#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time, itertools, itertools
from util import vector_slice, discount


def process_rollout_gae(rewards, values, gamma, lambda_=1.0, r=0.):
    """
    given a rollout, compute its returns and the advantage
    """

    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    rewards = np.asarray(rewards)
    vpred_t = np.asarray(values + [r])
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    rewards_plus_v = np.asarray(list(rewards) + [r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    return batch_r, batch_adv

def partial_rollout(env_reset, env_step, pi_v_h_func, zero_state, n_ticks=None):
    done = True
    tick = 0
    while True:
        rollout_start = time.time()
        # on-policy rollout
        if done:
            # reset episode stats
            episode_len = 0
            episode_reward = 0.

            # reset the env
            observation = env_reset()

            # initial rnn state
            h = zero_state
            h0 = h

        obs, actions, rewards, terminals, vhats, info = [], [], [], [], [], {}
        # sample some ticks for training
        for t in xrange(n_ticks) if n_ticks != None else itertools.count():
            obs.append(observation)
            # sample action according to policy
            action, v, h = pi_v_h_func(observation, h)
            actions.append(action)

            observation, reward, done = env_step(action)
            tick += 1

            rewards.append(reward)
            vhats.append(v)

            episode_reward += reward
            episode_len += 1

            terminals.append(done)

            if done:
                # stop rollout at the end of the episode
                break

        # yield partial rollout
        info['rollout_dt'] = time.time() - rollout_start
        info['tick'] = tick
        if done:
            # report episode stats
            info['episode_len'] = episode_len
            info['episode_reward'] = episode_reward

        info['initial_state'] = h0
        info['final_state'] = h
        h0 = h

        # note the `obs` sequence has one extra final element than others
        # the final observation can be used for bootstraping reward-to-go
        yield obs + [observation], actions, rewards, terminals, vhats, info


class A3C(object):
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
                build_model(env_spec['observation_shape'], env_spec['action_size'], n_rnn_dim=args.n_rnn_dim)
                self.global_tick = global_tick = tf.get_variable('global_tick', [], 'int32', trainable=False, initializer=tf.zeros_initializer)
                gv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # local only
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.obs_ph, self.initial_state_ph, action_logits, state_values, _, self.pi_v_h_func, self.v_func, self.zero_state = build_model(env_spec['observation_shape'], env_spec['action_size'], n_rnn_dim=args.n_rnn_dim)
                lv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            self.local_step = 0
            # copy parameters from `global/` to `local/``
            self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(lv, gv)])

            # define objectives
            self.actions_taken_ph = actions_taken_ph = tf.placeholder('int32')
            self.target_value_ph = target_value_ph = tf.placeholder('float')
            self.advantage_ph = advantage_ph = tf.placeholder('float')

            # entropy regularizer to encourage action diversity
            log_action_probs = tf.nn.log_softmax(action_logits)
            action_probs = tf.nn.softmax(action_logits)

            action_entropy = - tf.reduce_sum(action_probs * log_action_probs)
            action_logits = vector_slice(log_action_probs, actions_taken_ph)

            # objective for value estimation
            value_objective = tf.reduce_sum(tf.square(target_value_ph - state_values))

            # objective for computing policy gradient
            policy_objective = tf.reduce_sum(action_logits * advantage_ph)

            # total objective
            # maximize policy objective
            # minimize value objective
            # maximize action entropy
            objective = - policy_objective + args.value_objective_coeff * value_objective - args.action_entropy_coeff * action_entropy

            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            grads = tf.gradients(objective, lv)
            # apply gradients to the global parameters
            batch_len = tf.shape(self.obs_ph)[0]
            per_batch_len = 1. / tf.to_float(batch_len)
            inc_tick = global_tick.assign_add(batch_len)

            self.reward_gamma = args.reward_gamma
            self.td_lambda = args.td_lambda

            self.writer = writer

            # summaries
            self.episode_len_ph = episode_len_ph = tf.placeholder('float', name='episode_len')
            self.episode_reward_ph = episode_reward_ph = tf.placeholder('float', name='episode_reward')
            self.ticks_per_second_ph = ticks_per_second_ph = tf.placeholder('float', name='ticks_per_second')
            self.steps_per_second_ph = steps_per_second_ph = tf.placeholder('float', name='steps_per_second')

            self.per_episode_summary = tf.summary.merge([
                tf.summary.scalar('episodic/reward', episode_reward_ph),
                tf.summary.scalar('episodic/length', episode_len_ph),
                tf.summary.scalar('episodic/reward_per_tick', episode_reward_ph / episode_len_ph),
            ])

            norm = tf.global_norm(grads)
            var_norm = tf.global_norm(lv)

            # gradient clipping
            normed_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm, norm)
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
                    tf.summary.scalar('model/clipped_gradient_norm', tf.minimum(args.clip_norm, norm)),
                    tf.summary.scalar('model/var_norm', var_norm),
                    tf.summary.scalar('chief/steps_per_second', steps_per_second_ph),
                    tf.summary.scalar('chief/ticks_per_second', ticks_per_second_ph),
                ])

            self.rollout_generator = partial_rollout(env_reset, env_step, self.pi_v_h_func, zero_state=self.zero_state, n_ticks=args.n_update_ticks)
            self.step_start_at = None


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
        obs, actions, rewards, terminals, vhats, info = ro

        # bootstrap reward-to-go with state value estimate
        next_state_value = 0. if terminals[-1] else self.v_func(obs[-1], info['final_state'])
        rewards_to_go, advantages = process_rollout_gae(rewards, vhats, self.reward_gamma, self.td_lambda, next_state_value)

        delta_tick = len(actions)

        feed = {
            self.obs_ph: obs[:-1],
            self.initial_state_ph: info['initial_state'],
            self.target_value_ph: rewards_to_go,
            self.actions_taken_ph: actions,
            self.advantage_ph: advantages,
            self.steps_per_second_ph: 1. / step_dt,
            self.ticks_per_second_ph: delta_tick / info['rollout_dt'],
            }

        if self.is_chief and self.local_step % self.summary_interval == 0:
            per_episode_summary_val, _, gt = sess.run([self.per_step_summary, self.update_op, self.global_tick], feed_dict=feed)
            self.writer.add_summary(per_episode_summary_val, gt)
        else:
            _, gt = sess.run([self.update_op, self.global_tick], feed_dict=feed)

        self.local_step += 1

        if terminals[-1]:
            print 'tick', info['tick'], 'reward', info['episode_reward'], 'episode length', info['episode_len']

            per_episode_summary_val = self.per_episode_summary.eval({
                self.episode_reward_ph: info['episode_reward'],
                self.episode_len_ph: info['episode_len'],
            })
            self.writer.add_summary(per_episode_summary_val, gt)

def add_arguments(parser):
    parser.add_argument('--action-entropy-coeff', type=float, default=0.01)
    parser.add_argument('--value-objective-coeff', type=float, default=0.1)
    parser.add_argument('--reward-gamma', type=float, default=0.99)
    parser.add_argument('--td-lambda', type=float, default=1.)
    parser.add_argument('--n-update-ticks', type=int, default=20)
    parser.add_argument('--n-rnn-dim', type=int, default=256)
    parser.add_argument('--clip-norm', type=float, default=40.)
    parser.add_argument('--summary-interval', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)