#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time, argparse, itertools
from functools import partial
from util import vector_slice, discount, mask_slice, get_optimizer, mc_return, n_step_return, td_return, lambda_return
from core import Agent, StatefulAgent, Trainer
from gym import spaces

def lambda_advantage(rewards, values, gamma, td_lambda, bootstrap_value):
    td_advantages = td_return(rewards, values, gamma, bootstrap_value) - values
    # these terms telescope into lambda_advantage = G_t^lambda - V(S_t)
    lambda_advantages = discount(td_advantages, gamma * td_lambda)
    return lambda_advantages

class StatefulActorCriticAgent(StatefulAgent):
    def __init__(self, env_spec, build_model):
        assert isinstance(env_spec['action_space'], spaces.Discrete)

        tf_vars = build_model(env_spec['observation_space'].shape, env_spec['action_space'].n)
        assert len(tf_vars) == 6

        self.obs_ph, self.initial_state_ph, self.action_logits, self.state_values, self.final_state, self.zero_state = tf_vars
        self.actions = tf.squeeze(tf.multinomial(self.action_logits, 1), 1)

        self.spec = {
            'deterministic': False,
        }
        self._history_state = self.zero_state

    def pi_v(self, ob):
        sess = tf.get_default_session()
        action_val, state_value_val, next_history_state_val = sess.run([self.actions, self.state_values, self.final_state], {
            self.obs_ph: [ob],
            self.initial_state_ph: self._history_state,
        })
        self._history_state = next_history_state_val
        return action_val[0], state_value_val[0]

    def state_value(self, ob):
        return self.state_values.eval(feed_dict={
            self.obs_ph: [ob],
            self.initial_state_ph: self._history_state,
        })[0]

    def act(self, ob):
        sess = tf.get_default_session()
        action_val, next_history_state_val = sess.run([self.actions, self.final_state], {
            self.obs_ph: [ob],
            self.initial_state_ph: self._history_state,
        })
        self._history_state = next_history_state_val
        return action_val[0]

    def initial_state(self):
        return self.zero_state

    def history_state(self):
        return self._history_state

    def reset(self, history=None):
        if history:
            self._history_state = history
        else:
            self._history_state = self.zero_state

class ActorCriticAgent(Agent):
    def __init__(self, env_spec, build_model):
        assert isinstance(env_spec['action_space'], spaces.Discrete)

        tf_vars = build_model(env_spec['observation_space'].shape, env_spec['action_space'].n)
        assert len(tf_vars) == 3

        self.obs_ph, self.action_logits, self.state_values = tf_vars
        self.actions = tf.squeeze(tf.multinomial(self.action_logits, 1), 1)

        self.spec = {
            'deterministic': False,
        }

    def pi_v(self, ob):
        sess = tf.get_default_session()
        action_val, state_value_val = sess.run([self.actions, self.state_values], {
            self.obs_ph: [ob],
        })
        return action_val[0], state_value_val[0]

    def state_value(self, ob):
        return self.state_values.eval(feed_dict={
            self.obs_ph: [ob],
        })[0]

    def act(self, ob):
        return self.actions.eval(feed_dict={
            self.obs_ph: [ob],
        })[0]

class A3CTrainer(Trainer):
    ''' trainer for asynchronous advantage actor critic algorithm '''
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--action-entropy-coeff', type=float, default=0.01)
        parser.add_argument('--value-objective-coeff', type=float, default=0.1)
        parser.add_argument('--reward-gamma', type=float, default=0.99)

        parser.add_argument('--return-eval', choices=['td', 'mc', 'n-step', 'lambda'], default='mc')
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

        return parser

    def __init__(self, env, build_agent, task_index, writer, args):
        print '* A3C arguments:'
        vargs = vars(args)
        for k in sorted(vargs.keys()):
            print k, vargs[k]

        self.env = env
        self.task_index = task_index
        self.is_chief = task_index == 0

        # build compute graphs
        worker_device = '/job:worker/task:{}'.format(task_index)
        # on parameter server and locally
        # TODO clone the variables from local
        with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device=worker_device)):
            with tf.variable_scope('global'):
                build_agent(env.spec)
                global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                self.global_tick = tf.get_variable('global_tick', [], 'int32', trainable=False, initializer=tf.zeros_initializer())
                # shared the optimizer
                if args.shared:
                    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)

        # local only
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.agent = build_agent(env.spec)
                assert isinstance(self.agent, (ActorCriticAgent, StatefulActorCriticAgent))
                self.use_history = isinstance(self.agent, StatefulActorCriticAgent)
                local_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            self.local_step = 0
            # copy parameters from `global/` to `local/``
            self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(local_variables, global_variables)])

            # define objectives
            # inputs
            self.actions_taken_ph = tf.placeholder('int32')
            self.target_value_ph = tf.placeholder('float')
            self.advantage_ph = tf.placeholder('float')

            # entropy regularizer to encourage action diversity (naive exploration)
            log_action_probs = tf.nn.log_softmax(self.agent.action_logits)
            action_probs = tf.nn.softmax(self.agent.action_logits)

            self.action_entropy = - tf.reduce_sum(action_probs * log_action_probs)
            taken_action_logits = vector_slice(log_action_probs, self.actions_taken_ph)
            # taken_action_logits = mask_slice(log_action_probs, actions_taken_ph)

            # objective for value estimation
            self.value_objective = tf.reduce_sum(tf.square(self.target_value_ph - self.agent.state_values))

            # objective for computing policy gradient
            self.policy_objective = tf.reduce_sum(taken_action_logits * self.advantage_ph)

            # total objective
            # maximize policy objective
            # minimize value objective
            # maximize action entropy
            self.objective = - self.policy_objective + args.value_objective_coeff * self.value_objective - args.action_entropy_coeff * self.action_entropy

            grads = tf.gradients(self.objective, local_variables)
            # apply gradients to the global parameters
            batch_len = tf.shape(self.actions_taken_ph)[0]
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
            var_norm = tf.global_norm(local_variables)

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

            self.update_op = tf.group(optimizer.apply_gradients(zip(normed_grads, global_variables)), inc_tick)

            self.summary_interval = args.summary_interval
            if self.is_chief:
                print '* gradients'
                grad_summaries = []
                for g, v in zip(normed_grads, global_variables):
                    grad_summaries.append(tf.summary.histogram('gradients/%s' % v.name, g))
                    print '%s -> %s' % (g.name, v.name)

                self.per_step_summary = tf.summary.merge(grad_summaries + [
                    tf.summary.scalar('model/objective', self.objective * per_batch_len),
                    tf.summary.scalar('model/state_value_objective', self.value_objective * per_batch_len),
                    tf.summary.scalar('model/policy_objective', self.policy_objective * per_batch_len),
                    tf.summary.scalar('model/action_perplexity', tf.exp(self.action_entropy * per_batch_len)),
                    tf.summary.scalar('model/gradient_norm', norm),
                    tf.summary.scalar('model/clipped_gradient_norm', clipped_norm),
                    tf.summary.scalar('model/var_norm', var_norm),
                    tf.summary.scalar('chief/steps_per_second', self.steps_per_second_ph),
                    tf.summary.scalar('chief/ticks_per_second', self.ticks_per_second_ph),
                ])

            self.n_update_ticks = None if args.n_update_ticks == 0 else args.n_update_ticks

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

    def partial_rollout(self):
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
                observation = self.env.reset()

                # initial rnn state
                if self.use_history:
                    self.agent.reset()
                    h0 = self.agent.history_state()

            obs, actions, rewards, vhats, info = [], [], [], [], {}
            # sample some ticks for training
            for t in xrange(self.n_update_ticks) if self.n_update_ticks != None else itertools.count():
                obs.append(observation)
                # sample action according to policy
                action, vhat = self.agent.pi_v(observation)
                actions.append(action)

                observation, reward, done = self.env.step(action)

                tick += 1

                rewards.append(reward)
                vhats.append(vhat)

                episode_reward += reward
                episode_len += 1

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

            if self.use_history:
                info['initial_state'] = h0
                h0 = self.agent.history_state()

            # note the `obs` sequence has one extra final element than others
            # the final observation can be used for bootstraping reward-to-go
            yield np.asarray(obs + [observation]), np.asarray(actions), np.asarray(rewards), np.asarray(vhats), done, info

    def setup(self):
        self.rollout_generator = self.partial_rollout()
        if self.use_history:
            self.agent.reset()

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
        bootstrap_value = 0. if done else self.agent.state_value(obs[-1])

        returns = self.process_returns(rewards, vhats, bootstrap_value)
        advantages = self.process_advantages(rewards, vhats, bootstrap_value)

        feed = {
            self.agent.obs_ph: obs[:-1],
            self.target_value_ph: returns,
            self.actions_taken_ph: actions,
            self.advantage_ph: advantages,
            self.steps_per_second_ph: 1. / step_dt,
            self.ticks_per_second_ph: len(actions) / info['rollout_dt'],
            }

        if self.use_history:
            feed[self.agent.initial_state_ph] = info['initial_state']

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

def get_adv_ac_agent_builder(agent_id):
    parts = agent_id.split('.')
    if parts[0] == 'cnn_gru':
        from models.pi_v import cnn_gru
        n_cnn_layers = int(parts[1]) if len(parts) > 1 else 4
        n_cnn_filters = int(parts[2]) if len(parts) > 2 else 32
        n_rnn_dim = int(parts[3]) if len(parts) > 3 else 256
        return partial(StatefulActorCriticAgent, build_model=partial(cnn_gru.build_model, n_cnn_layers=n_cnn_layers, n_cnn_filters=n_cnn_filters, n_rnn_dim=n_rnn_dim))
    if parts[0] == 'cnn_lstm':
        from models.pi_v import cnn_lstm
        n_cnn_layers = int(parts[1]) if len(parts) > 1 else 4
        n_cnn_filters = int(parts[2]) if len(parts) > 2 else 32
        n_rnn_dim = int(parts[3]) if len(parts) > 3 else 256
        return partial(StatefulActorCriticAgent, build_model=partial(cnn_lstm.build_model, n_cnn_layers=n_cnn_layers, n_cnn_filters=n_cnn_filters, n_rnn_dim=n_rnn_dim))
    if parts[0] == 'deepmind':
        from models.pi_v import deepmind_cnn_lstm
        return partial(StatefulActorCriticAgent, build_model=deepmind_cnn_lstm.build_model)
    if parts[0] == 'ff_cnn':
        from models.pi_v import ff_cnn
        n_cnn_layers = int(parts[1]) if len(parts) > 1 else 4
        n_cnn_filters = int(parts[2]) if len(parts) > 2 else 32
        n_fc_dim = int(parts[3]) if len(parts) > 3 else 256
        return partial(ActorCriticAgent, build_model=partial(ff_cnn.build_model, n_cnn_layers=n_cnn_layers, n_cnn_filters=n_cnn_filters, n_fc_dim=n_fc_dim))
    if parts[0] == 'ff_fc':
        # feedforward
        from models.pi_v import ff_fc
        n_fc_layers = int(parts[1]) if len(parts) > 1 else 1
        n_fc_dim = int(parts[2]) if len(parts) > 2 else 32
        return partial(ActorCriticAgent, build_model=partial(ff_fc.build_model, n_fc_layers=n_fc_layers, n_fc_dim=n_fc_dim))
