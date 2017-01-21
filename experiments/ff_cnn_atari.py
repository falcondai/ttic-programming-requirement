#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import scipy.signal
import os, sys, time, json
import argparse
import importlib
import gym
from util import vector_slice, get_current_run_id, restore_vars


def discount(x, gamma):
    # return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    acc_reward = 0.
    partial_rewards_to_go = []
    for reward in x[::-1]:
        acc_reward = reward + gamma * acc_reward
        partial_rewards_to_go.insert(0, acc_reward)
    return np.asarray(partial_rewards_to_go)

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


def process_rollout(rewards, vhats, reward_gamma=0.99, next_state_value=0.):
    # reward_gamma < 1. prevents value function from being a constant function
    acc_reward = next_state_value
    partial_rewards_to_go = []
    for reward in rewards[::-1]:
        acc_reward = reward + reward_gamma * acc_reward
        partial_rewards_to_go.insert(0, acc_reward)

    vhats = np.asarray(list(vhats) + [next_state_value])
    # A(s, a) = Q(s, a) - V(s) = E[r + V(s') - V(s)]
    advantages = np.asarray(rewards) + reward_gamma * vhats[1:] - vhats[:-1]
    return partial_rewards_to_go, advantages

def partial_rollout(env, pi_v_func, zero_state=None, n_ticks=None):
    env_spec, env_step, env_reset, env_render = env

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

        if env_render != None:
            env_render()

        rewards.append(reward)
        vhats.append(v)

        episode_reward += reward
        episode_len += 1
        delta_tick += 1

        # enforce time limit
        if env_spec['timestep_limit'] != None and episode_len >= env_spec['timestep_limit']:
            done = True
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


def train(env, args, build_model):
    env_spec, env_step, env_reset, env_render = env

    summary_dir = 'tf-log/%s%d-%s' % (args['summary_prefix'], time.time(),
                                      os.path.basename(args['checkpoint_dir']))

    # set seeds
    np.random.seed(args['np_seed'])

    # create checkpoint dirs
    if not os.path.exists(args['checkpoint_dir']):
        os.makedirs(args['checkpoint_dir'])

    print '* training hyperparameters:'
    for k in sorted(args.keys()):
        print k, args[k]
    n_run = get_current_run_id(args['checkpoint_dir'])
    with open('%s/hyperparameters.%i.json' % (args['checkpoint_dir'], n_run),
              'wb') as hpf:
        json.dump(args, hpf)

    with tf.Graph().as_default() as g:
        # model
        print '* building model %s' % args['model']
        with tf.variable_scope('model'):
            obs_ph, _, action_logits, state_values, _, \
            pi_v_func, v_func, _ = build_model(env_spec['observation_shape'],
                                               env_spec['action_size'],
                                               n_cnn_layers=4,
                                               n_fc_dim=256)

        actions_taken_ph = tf.placeholder('int32')
        target_value_ph = tf.placeholder('float')
        advantage_ph = tf.placeholder('float')

        # entropy regularizer to encourage action diversity
        log_action_probs = tf.nn.log_softmax(action_logits)
        action_probs = tf.nn.softmax(action_logits)

        action_entropy = - tf.reduce_sum(action_probs * log_action_probs)
        action_logits = vector_slice(log_action_probs, actions_taken_ph)

        # objective for value estimation
        value_objective = tf.reduce_sum(tf.square(target_value_ph \
                                                  - state_values))

        # objective for computing policy gradient
        # state_advantage = target_value_ph - state_values
        policy_objective = tf.reduce_sum(action_logits * advantage_ph)

        # total objective
        # maximize policy objective and minimize value objective
        # and maximize action entropy
        objective = - policy_objective \
            + args['value_objective_coeff'] * value_objective \
            - args['action_entropy_coeff'] * action_entropy

        # optimization
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # global tick counts ticks experienced by the agent
        global_tick = tf.Variable(0, trainable=False, name='global_tick')
        delta_tick_ph = tf.placeholder('int32')
        increment_global_tick = global_tick.assign_add(delta_tick_ph)

        learning_rate = tf.train.exponential_decay(
            args['initial_learning_rate'],
            global_step, args['n_decay_steps'],
            args['decay_rate'],
            staircase=not args['no_decay_staircase'])

        if args['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate,
                                               args['adam_beta1'],
                                               args['adam_beta2'],
                                               args['adam_epsilon'])
        elif args['optimizer'] == 'ag':
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   args['momentum'],
                                                   use_nesterov=True)
        elif args['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                  args['rmsprop_decay'],
                                                  args['momentum'],
                                                  args['rmsprop_epsilon'])
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   args['momentum'])

        # train ops
        grad_vars = optimizer.compute_gradients(objective)
        update_op = optimizer.apply_gradients(grad_vars,
                                              global_step=global_step)

        # summaries
        episode_len_ph = tf.placeholder('float')
        episode_reward_ph = tf.placeholder('float')
        ticks_per_second_ph = tf.placeholder('float')
        steps_per_second_ph = tf.placeholder('float')
        # images_ph = tf.placeholder('uint8')

        batch_len = tf.to_float(tf.shape(obs_ph)[0])

        per_episode_summary = tf.summary.merge([
            tf.summary.scalar('episodic/reward', episode_reward_ph),
            tf.summary.scalar('episodic/length', episode_len_ph),
            tf.summary.scalar('episodic/reward_per_tick',
                              episode_reward_ph / episode_len_ph),
        ])

        grad_summaries = []
        grads = []
        var_list = []
        print '* extra summary'
        for g, v in grad_vars:
            grad_summaries.append(tf.summary.histogram('gradients/%s' % v.name, g))
            print 'gradients/%s' % v.name
            grads.append(g)
            var_list.append(v)

        normed_gn = norm = tf.global_norm(grads)
        var_norm = tf.global_norm(var_list)

        # gradient clipping
        normed_grads, norm = tf.clip_by_global_norm(grads, 40.)
        update_op = optimizer.apply_gradients(list(zip(normed_grads, var_list)),
                                              global_step=global_step)
        normed_gn = tf.global_norm(normed_grads)

        per_step_summary = tf.summary.merge(grad_summaries + [
            tf.summary.scalar('model/learning_rate', learning_rate),
            tf.summary.scalar('model/objective', objective / batch_len),
            tf.summary.scalar('model/state_value_objective',
                              value_objective / batch_len),
            tf.summary.scalar('model/policy_objective', policy_objective / batch_len),
            tf.summary.scalar('model/action_entropy', action_entropy / batch_len),
            tf.summary.scalar('model/action_perplexity', tf.exp(action_entropy / batch_len)),
            tf.summary.scalar('model/gradient_norm', norm),
            tf.summary.scalar('model/var_norm', var_norm),
            tf.summary.scalar('model/steps_per_second', steps_per_second_ph),
            # tf.image_summary('frames', images_ph, max_images=3),
            tf.summary.scalar('model/ticks_per_second', ticks_per_second_ph),
            tf.summary.scalar('model/gradient_norm_after_clip', normed_gn),
        ])

        saver = tf.train.Saver(max_to_keep=2, \
                               keep_checkpoint_every_n_hours=1)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(summary_dir, sess.graph,
                                           flush_secs=30)
            print '* writing summary to', summary_dir

            tf.train.export_meta_graph('%s/model.meta' % args['checkpoint_dir'])
            restore_vars(saver, sess, args['checkpoint_dir'], args['restart'])

            # print '* regularized parameters:'
            # for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            #     print v.name

            step = 0
            step_start = time.time()
            for ro in partial_rollout(env, pi_v_func,
                                      zero_state=None,
                                      n_ticks=args['n_update_ticks']):
                obs, actions, rewards, terminals, vhats, info = ro
                # print np.shape(obs), len(actions), terminals[-1], info

                # bootstrap reward-to-go with state value estimate
                next_state_value = 0. if terminals[-1] else v_func(obs[-1])
                # rewards_to_go, advantages = process_rollout(rewards, vhats, args['reward_gamma'], next_state_value)
                # rewards_to_go, advantages = process_rollout(rewards, vhats, args['reward_gamma'], 1., next_state_value)

                rewards_to_go1, advantages1 = process_rollout_gae(rewards, vhats, args['reward_gamma'], 1., next_state_value)

                # print next_state_value, rewards, rewards_to_go
                # print 'rtg', rewards_to_go, rewards_to_go1, rewards_to_go - rewards_to_go1
                # print 'adv', advantages, advantages1, advantages - advantages1

                delta_tick = len(actions)
                tick = info['tick']
                step_dt = time.time() - step_start
                step_start = time.time()

                per_step_summary_val, gt, _ = sess.run([per_step_summary, increment_global_tick, update_op], feed_dict={
                    obs_ph: obs[:-1],
                    target_value_ph: rewards_to_go1,
                    advantage_ph: advantages1,
                    actions_taken_ph: actions,
                    steps_per_second_ph: 1. / step_dt,
                    # images_ph: obs[-4:-1],
                    delta_tick_ph: delta_tick,
                    ticks_per_second_ph: delta_tick / info['rollout_dt'],
                    })

                writer.add_summary(per_step_summary_val, gt)

                if terminals[-1]:
                    per_episode_summary_val = per_episode_summary.eval({
                        episode_reward_ph: info['episode_reward'],
                        episode_len_ph: info['episode_len'],
                    })
                    writer.add_summary(per_episode_summary_val, gt)

                if step % args['n_save_interval'] == 0:
                    saver.save(sess, args['checkpoint_dir'] + '/model',
                               global_step=global_step.eval(), write_meta_graph=False)


def build_argparser():
    parse = argparse.ArgumentParser()

    parse.add_argument('--model', default='ff_cnn_pi_v')

    # gym options
    parse.add_argument('--env', default='PongDeterministic-v3')
    parse.add_argument('--monitor', action='store_true')
    parse.add_argument('--monitor_dir',
                       default='/tmp/gym-monitor-%i' % time.time())
    parse.add_argument('--scale', type=float, default=0.27)
    parse.add_argument('--interpolation', choices=['nearest', 'bilinear',
                                                   'bicubic', 'cubic'],
                       default='bilinear')

    parse.add_argument('--action_entropy_coeff', type=float, default=0.01)
    parse.add_argument('--value_objective_coeff', type=float, default=0.5)
    parse.add_argument('--reward_gamma', type=float, default=0.99)
    parse.add_argument('--dropout_rate', type=float, default=0.2)

    parse.add_argument('--restart', action='store_true')
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--summary_prefix', default='')
    parse.add_argument('--render', action='store_true')

    # how many episodes to rollout before update parameters
    parse.add_argument('--n_update_ticks', type=int, default=20)
    parse.add_argument('--n_save_interval', type=int, default=8)
    parse.add_argument('--n_eval_episodes', type=int, default=4)
    parse.add_argument('--n_eval_interval', type=int, default=8)

    # optimizer options
    parse.add_argument('--momentum', type=float, default=0.)
    parse.add_argument('--adam_beta1', type=float, default=0.9)
    parse.add_argument('--adam_beta2', type=float, default=0.999)
    parse.add_argument('--adam_epsilon', type=float, default=1e-8)
    parse.add_argument('--rmsprop_decay', type=float, default=0.9)
    parse.add_argument('--rmsprop_epsilon', type=float, default=1e-10)

    # training options
    parse.add_argument('--optimizer', choices=['adam', 'momentum', 'ag',
                                               'rmsprop'], default='adam')
    parse.add_argument('--initial_learning_rate', type=float, default=1e-4)
    parse.add_argument('--n_decay_steps', type=int, default=512)
    parse.add_argument('--no_decay_staircase', action='store_true')
    parse.add_argument('--decay_rate', type=float, default=1.)

    parse.add_argument('--np_seed', type=int, default=123)

    return parse


if __name__ == '__main__':
    from functools import partial
    from envs.core import GymEnv
    from envs.wrappers import GrayscaleWrapper, StackFrameWrapper, ScaleWrapper, KeyMapWrapper

    # arguments
    parse = build_argparser()
    args = parse.parse_args()

    gym_env = GymEnv('PongDeterministic-v3')
    env = StackFrameWrapper(GrayscaleWrapper(ScaleWrapper(gym_env, 42./160.)), 4)

    env_render = env.render if args.render else None

    print '* environment', args.env
    print 'observation shape', env.spec['observation_shape']
    print 'action space', env.spec['action_size']
    print 'timestep limit', env.spec['timestep_limit']

    # model
    model = importlib.import_module('models.%s' % args.model)

    # train
    # gym monitor
    if args.monitor:
        env.monitor.start(args['monitor_dir'])

    train((env.spec, env.step, env.reset, env_render),
          vars(args),
          model.build_model)

    if args.monitor:
        env.monitor.close()
