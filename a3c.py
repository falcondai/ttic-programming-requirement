#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, itertools, json
from Queue import deque
import tqdm
import argparse
import importlib
import gym
from util import vector_slice, get_current_run_id, restore_vars, discount


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

# def process_rollout(rewards, vhats, reward_gamma=0.99, next_state_value=0.):
#     # reward_gamma < 1. prevents value function from being a constant function
#     acc_reward = next_state_value
#     partial_rewards_to_go = []
#     for reward in rewards[::-1]:
#         acc_reward = reward + reward_gamma * acc_reward
#         partial_rewards_to_go.insert(0, acc_reward)
#
#     vhats = np.asarray(list(vhats) + [next_state_value])
#     # A(s, a) = Q(s, a) - V(s) = E[r + V(s') - V(s)]
#     advantages = np.asarray(rewards) + reward_gamma * vhats[1:] - vhats[:-1]
#     return partial_rewards_to_go, advantages

def partial_rollout(env_reset, env_step, pi_v_func, zero_state=None, n_ticks=None):
    done = True
    delta_tick = 0
    tick = 0
    while True:
        # on-policy rollout
        if done:
            obs, actions, rewards, terminals, vhats, info = [], [], [], [], [], {}
            rollout_start = time.time()

            # reset the env
            observation = env_reset()

            if zero_state != None:
                # initial state for pi_func
                h = zero_state
                h0 = h

            # reset episode stats
            episode_len = 0
            episode_reward = 0.

        obs.append(observation)
        # sample action according to policy
        if zero_state != None:
            action, v, h = pi_v_func(observation, h)
        else:
            action, v = pi_v_func(observation)
        actions.append(action)

        observation, reward, done = env_step(action)
        tick += 1

        rewards.append(reward)
        vhats.append(v)

        episode_reward += reward
        episode_len += 1
        delta_tick += 1

        terminals.append(done)

        # yield partial rollout
        if done or (n_ticks != None and delta_tick == n_ticks):
            # got enough ticks for training
            # note the `obs` sequence has one extra element than others
            # the final observation can be used for bootstraping reward-to-go
            info['rollout_dt'] = time.time() - rollout_start
            info['tick'] = tick
            if done:
                # report episode stats
                info['episode_len'] = episode_len
                info['episode_reward'] = episode_reward

            if zero_state != None:
                info['initial_state'] = h0
                info['final_state'] = h

            yield obs + [observation], actions, rewards, terminals, vhats, \
            info
            rollout_start = time.time()
            delta_tick = 0
            obs, actions, rewards, terminals, vhats, info = [], [], [], [], [], {}
            if not done and zero_state != None:
                h0 = h

class A3C(object):
    def __init__(self, env_spec, env_reset, env_step, build_model, task_index, writer, args):
        print '* training hyperparameters:'
        for k in sorted(args.keys()):
            print k, args[k]

        self.task_index = task_index

        # build compute graph
        worker_device = '/job:worker/task:{}/cpu:0'.format(task_index)
        with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device=worker_device)):
            with tf.variable_scope('global'):
                build_model(env_spec['observation_shape'], env_spec['action_size'], n_rnn_dim=256)
                self.global_tick = global_tick = tf.get_variable('global_tick', [], 'int32', trainable=False, initializer=tf.zeros_initializer)
                gv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.obs_ph, self.initial_state_ph, action_logits, state_values, _, \
                self.pi_v_func, self.v_func, self.zero_state = build_model(env_spec['observation_shape'], env_spec['action_size'], n_rnn_dim=256)
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
            objective = - policy_objective + args['value_objective_coeff'] * value_objective - args['action_entropy_coeff'] * action_entropy

            optimizer = tf.train.AdamOptimizer(1e-4)
            grads = tf.gradients(objective, lv)
            # apply gradients to the global parameters
            batch_len = tf.shape(self.obs_ph)[0]
            per_batch_len = 1. / tf.to_float(batch_len)
            inc_tick = global_tick.assign_add(batch_len)
            self.update_op = tf.group(optimizer.apply_gradients(zip(grads, gv)), inc_tick)

            self.reward_gamma = args['reward_gamma']
            self.td_lambda = args['td_lambda']

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
            # normed_grads, norm = tf.clip_by_global_norm(grads, 100.)
            # update_op = optimizer.apply_gradients(list(zip(normed_grads, var_list)),
            #                                       global_step=global_step)
            # normed_gn = tf.global_norm(normed_grads)

            self.per_step_summary = tf.summary.merge([
                # tf.summary.scalar('model/learning_rate', learning_rate),
                tf.summary.scalar('model/objective', objective * per_batch_len),
                tf.summary.scalar('model/state_value_objective', value_objective * per_batch_len),
                tf.summary.scalar('model/policy_objective', policy_objective * per_batch_len),
                tf.summary.scalar('model/action_entropy', action_entropy * per_batch_len),
                tf.summary.scalar('model/action_perplexity', tf.exp(action_entropy * per_batch_len)),
                tf.summary.scalar('model/gradient_norm', norm),
                tf.summary.scalar('model/var_norm', var_norm),
                tf.summary.scalar('model/steps_per_second', steps_per_second_ph),
                tf.summary.scalar('model/ticks_per_second', ticks_per_second_ph),
                # tf.summary.scalar('model/gradient_norm_after_clip', normed_gn),
            ])

            self.rollout_generator = partial_rollout(env_reset, env_step, self.pi_v_func, zero_state=self.zero_state, n_ticks=20)


    def train(self, sess):
        # # optimization
        # global_step = tf.Variable(0, trainable=False, name='global_step')
        # # global tick counts ticks experienced by the agent
        # global_tick = tf.Variable(0, trainable=False, name='global_tick')
        # delta_tick_ph = tf.placeholder('int32')
        # increment_global_tick = global_tick.assign_add(delta_tick_ph)
        #
        # # train ops
        # grad_vars = optimizer.compute_gradients(objective)
        # update_op = optimizer.apply_gradients(grad_vars,
        #                                       global_step=global_step)
        #
        # # summaries
        # episode_len_ph = tf.placeholder('float')
        # episode_reward_ph = tf.placeholder('float')
        # ticks_per_second_ph = tf.placeholder('float')
        # steps_per_second_ph = tf.placeholder('float')
        #
        # batch_len = tf.to_float(tf.shape(obs_ph)[0])
        #
        # # per_episode_summary = tf.summary.merge([
        # #     tf.summary.scalar('episodic/reward', episode_reward_ph),
        # #     tf.summary.scalar('episodic/length', episode_len_ph),
        # #     tf.summary.scalar('episodic/reward_per_tick',
        # #                       episode_reward_ph / episode_len_ph),
        # # ])
        #
        # grad_summaries = []
        # grads = []
        # var_list = []
        # print '* extra summary'
        # for g, v in grad_vars:
        #     # grad_summaries.append(tf.summary.histogram('gradients/%s' % v.name, g))
        #     print 'gradients/%s' % v.name
        #     grads.append(g)
        #     var_list.append(v)

        # norm = tf.global_norm(grads)
        # var_norm = tf.global_norm(var_list)

        # gradient clipping
        # normed_grads, norm = tf.clip_by_global_norm(grads, 100.)
        # update_op = optimizer.apply_gradients(list(zip(normed_grads, var_list)),
        #                                       global_step=global_step)
        # normed_gn = tf.global_norm(normed_grads)

        # per_step_summary = tf.summary.merge(grad_summaries + [
        #     tf.summary.scalar('model/learning_rate', learning_rate),
        #     tf.summary.scalar('model/objective', objective / batch_len),
        #     tf.summary.scalar('model/state_value_objective',
        #                       value_objective / batch_len),
        #     tf.summary.scalar('model/policy_objective', policy_objective / batch_len),
        #     tf.summary.scalar('model/action_entropy', action_entropy / batch_len),
        #     tf.summary.scalar('model/action_perplexity', tf.exp(action_entropy / batch_len)),
        #     tf.summary.scalar('model/gradient_norm', norm),
        #     tf.summary.scalar('model/var_norm', var_norm),
        #     tf.summary.scalar('model/steps_per_second', steps_per_second_ph),
        #     tf.summary.scalar('model/ticks_per_second', ticks_per_second_ph),
        #     tf.summary.scalar('model/gradient_norm_after_clip', normed_gn),
        # ])

        step_start = time.time()
        # synchronize with parameter server
        sess.run(self.sync_op)
        ro = self.rollout_generator.next()
        obs, actions, rewards, terminals, vhats, info = ro

        # bootstrap reward-to-go with state value estimate
        next_state_value = 0. if terminals[-1] else self.v_func(obs[-1], info['final_state'])
        rewards_to_go, advantages = process_rollout_gae(rewards, vhats, self.reward_gamma, self.td_lambda, next_state_value)

        delta_tick = len(actions)
        tick = info['tick']
        step_dt = time.time() - step_start
        step_start = time.time()

        should_fetch_summary = self.task_index == 0 and self.local_step % 16 == 0

        if should_fetch_summary:
            fetches = [self.per_step_summary, self.update_op, self.global_tick]
        else:
            fetches = [self.update_op, self.global_tick]

        fetched = sess.run(fetches, feed_dict={
            self.obs_ph: obs[:-1],
            self.initial_state_ph: info['initial_state'],
            self.target_value_ph: rewards_to_go,
            self.actions_taken_ph: actions,
            self.advantage_ph: advantages,
            self.steps_per_second_ph: 1. / step_dt,
            self.ticks_per_second_ph: delta_tick / info['rollout_dt'],
            })
        self.local_step += 1

        gt = fetched[-1]
        if should_fetch_summary:
            self.writer.add_summary(fetched[0], gt)

        if terminals[-1]:
            print 'reward', info['episode_reward'], 'episode length', info['episode_len']

            per_episode_summary_val = self.per_episode_summary.eval({
                self.episode_reward_ph: info['episode_reward'],
                self.episode_len_ph: info['episode_len'],
            })
            self.writer.add_summary(per_episode_summary_val, gt)
