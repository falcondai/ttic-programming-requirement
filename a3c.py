#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time
from util import vector_slice, discount, partial_rollout, mask_slice, get_optimizer, mc_return, n_step_return, td_return, lambda_return

def lambda_advantage(rewards, values, gamma, td_lambda, bootstrap_value):
    td_advantage = td_return(rewards, values, gamma, bootstrap_value) - values
    # these terms telescope into lambda_advantage = G_t^lambda - V(S_t)
    lambda_advantage = discount(td_advantage, gamma * td_lambda)
    return lambda_advantage

class A3C(object):
    ''' trainer for asynchronous advantage actor critic algorithm '''
    def __init__(self, env_spec, env_reset, env_step, build_model, task_index, writer, args):
        print '* A3C arguments:'
        vargs = vars(args)
        for k in sorted(vargs.keys()):
            print k, vargs[k]

        self.task_index = task_index
        self.is_chief = task_index == 0

        # build compute graphs
        worker_device = '/job:worker/task:{}'.format(task_index)
        # on parameter server and locally
        with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device=worker_device)):
            with tf.variable_scope('global'):
                build_model(env_spec['observation_shape'], env_spec['action_size'])
                self.global_tick = tf.get_variable('global_tick', [], 'int32', trainable=False, initializer=tf.zeros_initializer)
                # shared the optimizer
                if args.shared:
                    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
                gv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # local only
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.obs_ph, self.initial_state_ph, action_logits, state_values, _, self.pi_v_h_func, self.v_func, self.zero_state = build_model(env_spec['observation_shape'], env_spec['action_size'])
                lv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            self.local_step = 0
            # copy parameters from `global/` to `local/``
            self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(lv, gv)])

            # define objectives
            self.actions_taken_ph = tf.placeholder('int32')
            self.target_value_ph = tf.placeholder('float')
            self.advantage_ph = tf.placeholder('float')

            # entropy regularizer to encourage action diversity
            log_action_probs = tf.nn.log_softmax(action_logits)
            action_probs = tf.nn.softmax(action_logits)

            action_entropy = - tf.reduce_sum(action_probs * log_action_probs)
            taken_action_logits = vector_slice(log_action_probs, self.actions_taken_ph)
            # taken_action_logits = mask_slice(log_action_probs, actions_taken_ph)

            # objective for value estimation
            value_objective = tf.reduce_sum(tf.square(self.target_value_ph - state_values))

            # objective for computing policy gradient
            policy_objective = tf.reduce_sum(taken_action_logits * self.advantage_ph)

            # total objective
            # maximize policy objective
            # minimize value objective
            # maximize action entropy
            objective = - policy_objective + args.value_objective_coeff * value_objective - args.action_entropy_coeff * action_entropy

            grads = tf.gradients(objective, lv)
            # apply gradients to the global parameters
            batch_len = tf.shape(self.obs_ph)[0]
            per_batch_len = 1. / tf.to_float(batch_len)
            inc_tick = self.global_tick.assign_add(batch_len)

            self.reward_gamma = args.reward_gamma
            self.return_lambda = args.return_lambda
            self.return_n_step = args.return_n_step
            self.advantage_lambda = args.advantage_lambda
            self.advantage_n_step = args.advantage_n_step

            self.writer = writer

            # summaries
            self.episode_len_ph = tf.placeholder('float', name='episode_len')
            self.episode_reward_ph = tf.placeholder('float', name='episode_reward')
            self.ticks_per_second_ph = tf.placeholder('float', name='ticks_per_second')
            self.steps_per_second_ph = tf.placeholder('float', name='steps_per_second')

            self.per_episode_summary = tf.summary.merge([
                tf.summary.scalar('episodic/reward', self.episode_reward_ph),
                tf.summary.scalar('episodic/length', self.episode_len_ph),
                tf.summary.scalar('episodic/reward_per_tick', self.episode_reward_ph / self.episode_len_ph),
            ])

            norm = tf.global_norm(grads)
            var_norm = tf.global_norm(lv)

            # local optimizer
            if not args.shared:
                optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)

            if args.no_grad_clip:
                normed_grads = grads
                clipped_norm = norm
            else:
                # gradient clipping
                normed_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm, norm)
                clipped_norm = tf.minimum(args.clip_norm, norm)

            self.update_op = tf.group(optimizer.apply_gradients(zip(normed_grads, gv)), inc_tick)

            self.summary_interval = args.summary_interval
            if self.is_chief:
                grad_summaries = []
                for g, v in zip(normed_grads, gv):
                    grad_summaries.append(tf.summary.histogram('gradients/%s' % v.name, g))
                    print 'gradients/%s' % v.name

                self.per_step_summary = tf.summary.merge(grad_summaries + [
                    tf.summary.scalar('model/objective', objective * per_batch_len),
                    tf.summary.scalar('model/state_value_objective', value_objective * per_batch_len),
                    tf.summary.scalar('model/policy_objective', policy_objective * per_batch_len),
                    tf.summary.scalar('model/action_perplexity', tf.exp(action_entropy * per_batch_len)),
                    tf.summary.scalar('model/gradient_norm', norm),
                    tf.summary.scalar('model/clipped_gradient_norm', clipped_norm),
                    tf.summary.scalar('model/var_norm', var_norm),
                    tf.summary.scalar('chief/steps_per_second', self.steps_per_second_ph),
                    tf.summary.scalar('chief/ticks_per_second', self.ticks_per_second_ph),
                ])

            n_update_ticks = None if args.n_update_ticks == 0 else args.n_update_ticks

            self.rollout_generator = partial_rollout(env_reset, env_step, self.pi_v_h_func, zero_state=self.zero_state, n_ticks=n_update_ticks)
            self.step_start_at = None

            # process returns and advantages
            if args.return_eval == 'td':
                self.process_returns = lambda rewards, values, bootstrap_value: td_return(rewards, values, self.reward_gamma, bootstrap_value)
            elif args.return_eval == 'mc':
                self.process_returns = lambda rewards, values, bootstrap_value: mc_return(rewards, self.reward_gamma, bootstrap_value)
            elif args.return_eval == 'n-step':
                self.process_returns = lambda rewards, values, bootstrap_value: n_step_return(rewards, values, self.reward_gamma, bootstrap_value, self.return_n_step)
            else:
                self.process_returns = lambda rewards, values, bootstrap_value: lambda_return(rewards, values, self.reward_gamma, self.return_lambda, bootstrap_value)

            if args.advantage_eval == 'td':
                self.process_advantages = lambda rewards, values, bootstrap_value: td_return(rewards, values, self.reward_gamma, bootstrap_value) - values
            elif args.advantage_eval == 'mc':
                self.process_advantages = lambda rewards, values, bootstrap_value: mc_return(rewards, self.reward_gamma, bootstrap_value) - values
            elif args.advantage_eval == 'n-step':
                self.process_advantages = lambda rewards, values, bootstrap_value: n_step_return(rewards, values, self.reward_gamma, bootstrap_value, self.advantage_n_step) - values
            else:
                self.process_advantages = lambda rewards, values, bootstrap_value: lambda_advantage(rewards, values, self.reward_gamma, self.advantage_lambda, bootstrap_value)

    def train(self, sess):
        if self.step_start_at != None:
            step_dt = time.time() - self.step_start_at
        else:
            step_dt = 1.
        self.step_start_at = time.time()

        # synchronize with parameter server
        sess.run(self.sync_op)

        # sample a partial rollout
        ro = self.rollout_generator.next()
        obs, actions, rewards, vhats, done, info = ro

        # bootstrap returns with state value estimate
        bootstrap_value = 0. if done else self.v_func(obs[-1], info['final_state'])
        returns = self.process_returns(rewards, vhats, bootstrap_value)
        advantages = self.process_advantages(rewards, vhats, bootstrap_value)

        feed = {
            self.obs_ph: obs[:-1],
            self.target_value_ph: returns,
            self.actions_taken_ph: actions,
            self.advantage_ph: advantages,
            self.steps_per_second_ph: 1. / step_dt,
            self.ticks_per_second_ph: len(actions) / info['rollout_dt'],
            }

        # for RNN models
        if self.initial_state_ph != None:
            feed[self.initial_state_ph] = info['initial_state']

        if self.is_chief and self.local_step % self.summary_interval == 0:
            per_episode_summary_val, _, gt = sess.run([self.per_step_summary, self.update_op, self.global_tick], feed_dict=feed)
            self.writer.add_summary(per_episode_summary_val, gt)
        else:
            _, gt = sess.run([self.update_op, self.global_tick], feed_dict=feed)

        self.local_step += 1

        if done:
            print 'worker', self.task_index, 'tick', info['tick'], 'reward', info['episode_reward'], 'episode length', info['episode_len']

            per_episode_summary_val = self.per_episode_summary.eval({
                self.episode_reward_ph: info['episode_reward'],
                self.episode_len_ph: info['episode_len'],
            })
            self.writer.add_summary(per_episode_summary_val, gt)

def add_arguments(parser):
    parser.add_argument('--action-entropy-coeff', type=float, default=0.01)
    parser.add_argument('--value-objective-coeff', type=float, default=0.1)
    parser.add_argument('--reward-gamma', type=float, default=0.99)

    parser.add_argument('--return-eval', choices=['td', 'mc', 'n-step', 'lambda'], default='td')
    parser.add_argument('--return-n-step', type=int, default=10)
    parser.add_argument('--return-lambda', type=float, default=0.5)

    parser.add_argument('--advantage-eval', choices=['td', 'mc', 'n-step', 'lambda'], default='td')
    parser.add_argument('--advantage-n-step', type=int, default=10)
    parser.add_argument('--advantage-lambda', type=float, default=0.5)

    parser.add_argument('--n-update-ticks', type=int, default=20, help='update batch size, 0 for full episodes')
    parser.add_argument('--no-grad-clip', action='store_true', help='disable gradient clipping')
    parser.add_argument('--clip-norm', type=float, default=40.)
    parser.add_argument('--summary-interval', type=int, default=16)
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop', 'momentum'], default='adam')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--shared', action='store_true')
