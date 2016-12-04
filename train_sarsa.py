#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools, json
import tqdm
import argparse
import importlib
import gym
from util import vector_slice, rollout, pad_zeros, duplicate_obs,\
    to_epsilon_greedy

def get_current_run_id(checkpoint_dir):
    paths = glob.glob('%s/hyperparameters.*.json' % checkpoint_dir)
    if len(paths) == 0:
        return 0
    return sorted(map(lambda p: int(p.split('.')[-2]), paths))[-1] + 1

def restore_vars(saver, sess, checkpoint_dir, restart=False):
    '''Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage.'''
    sess.run(tf.initialize_all_variables())

    if not restart:
        path = tf.train.latest_checkpoint(checkpoint_dir)
        if path is None:
            print '* no existing checkpoint found'
            return False
        else:
            print '* restoring from %s' % path
            saver.restore(sess, path)
            return True
    print '* overwriting checkpoints at %s' % checkpoint_dir
    return False

def load_rollout(data_dir):
    pass

def train_q(env_spec, env_step, env_reset, env_render, args, build_q_model):
    summary_dir = 'tf-log/%s%d-%s' % (args['summary_prefix'], time.time(),
                                      os.path.basename(args['checkpoint_dir']))

    # set seeds
    np.random.seed(args['np_seed'])
    tf.set_random_seed(args['tf_seed'])

    # create checkpoint dirs
    if not os.path.exists(args['checkpoint_dir']):
        try:
            os.makedirs(args['checkpoint_dir'])
        except OSError:
            pass

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
        policy_input_shape = list(env_spec['observation_shape'])
        policy_input_shape[-1] *= args['n_obs_ticks']

        current_q_model_scope = 'current_q_model'
        with tf.variable_scope(current_q_model_scope):
            obs_ph, keep_prob_ph, action_values = build_q_model(
                policy_input_shape,
                env_spec['action_size'])

        with tf.variable_scope(current_q_model_scope, reuse=True):
            next_obs_ph, _, next_action_values = build_q_model(
                policy_input_shape,
                env_spec['action_size'])

        action_ph = tf.placeholder('int32')
        reward_ph = tf.placeholder('float')
        next_action_ph = tf.placeholder('int32')
        nonterminal_ph = tf.placeholder('float')

        avg_len_episode_ph = tf.placeholder('float')
        avg_episode_reward_ph = tf.placeholder('float')
        max_episode_reward_ph = tf.placeholder('float')
        min_episode_reward_ph = tf.placeholder('float')
        avg_tick_reward_ph = tf.placeholder('float')
        avg_objective_ph = tf.placeholder('float')
        epsilon_ph = tf.placeholder('float')

        # SARSA
        # r + gamma * Q(s', a'), where s', a' are the observed
        # according to behavior policy
        target = reward_ph + nonterminal_ph * args['reward_gamma'] \
            * vector_slice(next_action_values, next_action_ph)

        # action values over observed Q(s, a)
        Q_sa = vector_slice(action_values, action_ph)

        # violation of the consistency of Q as objective
        objective = tf.reduce_sum(tf.square(tf.stop_gradient(target) - Q_sa))

        # optimization
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(
            args['initial_learning_rate'],
            global_step, args['n_lr_decay_steps'],
            args['lr_decay_rate'],
            staircase=not args['no_lr_decay_staircase'])

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
        update_q_op = optimizer.apply_gradients(
            grad_vars,
            global_step=global_step)

        # summary
        if not args['no_summary']:
            tf.scalar_summary('learning_rate', learning_rate)
            tf.scalar_summary('average_episode_reward',
                              avg_episode_reward_ph)
            tf.scalar_summary('max_episode_reward', max_episode_reward_ph)
            tf.scalar_summary('min_episode_reward', min_episode_reward_ph)
            tf.scalar_summary('average_tick_reward', avg_tick_reward_ph)
            tf.scalar_summary('average_episode_length', avg_len_episode_ph)
            tf.scalar_summary('average_objective', avg_objective_ph)
            tf.scalar_summary('epsilon', epsilon_ph)

            print '* extra summary'
            for g, v in grad_vars:
                tf.histogram_summary('gradients/%s' % v.name, g)
                print 'gradients/%s' % v.name

            summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(max_to_keep=2,
                               keep_checkpoint_every_n_hours=1)
        with tf.Session() as sess:
            if not args['no_summary']:
                writer = tf.train.SummaryWriter(summary_dir, sess.graph,
                                                flush_secs=30)
                print '* writing summary to', summary_dir
            restore_vars(saver, sess, args['checkpoint_dir'], args['restart'])

            print '* regularized parameters:'
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                print v.name

            # epsilon-greedy policy based on current Q
            policy = lambda epsilon, obs: to_epsilon_greedy(
                epsilon,
                lambda _obs: action_values.eval(feed_dict={
                     obs_ph: [_obs],
                     keep_prob_ph: 1. - args['dropout_rate'],
                })[0],
                obs)

            for i in tqdm.tqdm(xrange(args['n_train_steps'])):
                # on-policy rollout for some episodes
                episodes = []
                n_ticks = 0
                episode_rewards = []
                epsilon = args['initial_epsilon'] / (1. + args['epsilon_decay_rate'] * global_step.eval())

                # sample rollouts
                for j in xrange(args['n_update_episodes']):
                    observations, actions, rewards = rollout(
                        partial(policy, epsilon),
                        env_spec,
                        env_step,
                        env_reset,
                        env_render,
                        n_obs_ticks=args['n_obs_ticks'],
                    )
                    episodes.append((observations, actions, rewards))
                    n_ticks += len(observations)
                    episode_rewards.append(np.sum(rewards))

                avg_len_episode = n_ticks * 1. / args['n_update_episodes']
                avg_tick_reward = np.sum(episode_rewards) * 1. / n_ticks

                # transform and preprocess the rollouts
                obs = []
                action_inds = []
                all_rewards = []
                nonterminals = []
                next_obs = []
                next_action_inds = []

                # process rollouts
                for observations, actions, rewards in episodes:
                    len_episode = len(observations)
                    dup_obs = list(duplicate_obs(pad_zeros(observations,
                                                        args['n_obs_ticks']),
                                              args['n_obs_ticks']))
                    obs += dup_obs
                    action_inds += actions
                    all_rewards += rewards
                    nonterminals += [1.] * (len_episode - 1) + [0.]
                    # pad zeros at the terminal tick
                    next_obs += dup_obs[1:] + [np.zeros(policy_input_shape)]
                    next_action_inds += actions[1:] + [0]

                # estimate and accumulate gradients by batches
                acc_obj_val = 0.
                acc_grads = dict([(grad, np.zeros(grad.get_shape()))
                                      for grad, var in grad_vars])
                n_batch = int(np.ceil(n_ticks * 1. / args['n_batch_ticks']))
                for j in xrange(n_batch):
                    start = j * args['n_batch_ticks']
                    end = min(start + args['n_batch_ticks'], n_ticks)
                    grad_feed = {
                        keep_prob_ph: 1. - args['dropout_rate'],
                        obs_ph: obs[start:end],
                        action_ph: action_inds[start:end],
                        reward_ph: all_rewards[start:end],
                        next_obs_ph: next_obs[start:end],
                        next_action_ph: next_action_inds[start:end],
                        nonterminal_ph: nonterminals[start:end],
                    }

                    # sum up gradients
                    obj_val, grad_vars_val = sess.run([
                        objective,
                        grad_vars,
                        ], feed_dict=grad_feed)
                    for (g, _), (g_val, _) in zip(grad_vars, grad_vars_val):
                        acc_grads[g] += g_val * (end - start) / n_ticks
                    acc_obj_val += obj_val

                # update current Q model
                update_dict = {
                    avg_len_episode_ph: avg_len_episode,
                    avg_episode_reward_ph: np.mean(episode_rewards),
                    max_episode_reward_ph: np.max(episode_rewards),
                    min_episode_reward_ph: np.min(episode_rewards),
                    avg_tick_reward_ph: avg_tick_reward,
                    avg_objective_ph: acc_obj_val / n_ticks,
                    epsilon_ph: epsilon,
                }
                update_dict.update(acc_grads)
                summary_val, _ = sess.run([summary_op, update_q_op],
                                          feed_dict=update_dict)

                if not args['no_summary']:
                    writer.add_summary(summary_val, global_step.eval())

                if i % args['n_save_interval'] == 0:
                    saver.save(sess, args['checkpoint_dir'] + '/model',
                               global_step=global_step.eval())

            # save again at the end
            saver.save(sess, args['checkpoint_dir'] + '/model',
                       global_step=global_step.eval())

def build_argparser():
    parse = argparse.ArgumentParser()

    parse.add_argument('--model', required=True)

    # gym options
    parse.add_argument('--env', default='CartPole-v0')
    parse.add_argument('--monitor', action='store_true')
    parse.add_argument('--monitor_dir',
                       default='/tmp/gym-monitor-%i' % time.time())
    parse.add_argument('--n_obs_ticks', type=int, default=1)
    parse.add_argument('--timestep_limit', type=int, default=10**9)
    parse.add_argument('--use_render_state', action='store_true')
    parse.add_argument('--scale', type=float, default=1.)
    parse.add_argument('--interpolation', choices=['nearest', 'bilinear',
                                                   'bicubic', 'cubic'],
                       default='nearest')

    # objective options
    parse.add_argument('--reg_coeff', type=float, default=0.0001)
    parse.add_argument('--reward_gamma', type=float, default=1.)
    parse.add_argument('--dropout_rate', type=float, default=0.2)

    parse.add_argument('--restart', action='store_true')
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--no_summary', action='store_true')
    parse.add_argument('--summary_prefix', default='')
    parse.add_argument('--render', action='store_true')

    # how many episodes to rollout before update parameters
    parse.add_argument('--n_update_episodes', type=int, default=4)
    parse.add_argument('--n_batch_ticks', type=int, default=128)
    parse.add_argument('--n_save_interval', type=int, default=1)
    parse.add_argument('--n_train_steps', type=int, default=10**5)
    parse.add_argument('--n_update_target_interval', type=int, default=4)

    # optimizer options
    parse.add_argument('--momentum', type=float, default=0.2)
    parse.add_argument('--adam_beta1', type=float, default=0.9)
    parse.add_argument('--adam_beta2', type=float, default=0.999)
    parse.add_argument('--adam_epsilon', type=float, default=1e-8)
    parse.add_argument('--rmsprop_decay', type=float, default=0.9)
    parse.add_argument('--rmsprop_epsilon', type=float, default=1e-10)

    # training options
    parse.add_argument('--optimizer', choices=['adam', 'momentum', 'ag',
                                               'rmsprop'], default='rmsprop')
    parse.add_argument('--initial_learning_rate', type=float, default=0.001)
    parse.add_argument('--n_lr_decay_steps', type=int, default=512)
    parse.add_argument('--no_lr_decay_staircase', action='store_true')
    parse.add_argument('--lr_decay_rate', type=float, default=0.8)
    parse.add_argument('--initial_epsilon', type=float, default=0.1)
    parse.add_argument('--epsilon_decay_rate', type=float, default=0.001)

    parse.add_argument('--np_seed', type=int, default=123)
    parse.add_argument('--tf_seed', type=int, default=1234)

    return parse


if __name__ == '__main__':
    from functools import partial
    from util import passthrough, use_render_state

    # arguments
    parse = build_argparser()
    args = parse.parse_args()

    gym_env = gym.make(args.env)
    if args.use_render_state:
        env_spec, env_step, env_reset, env_render = use_render_state(
            gym_env, args.scale, args.interpolation)
    else:
        env_spec, env_step, env_reset, env_render = passthrough(gym_env)

    env_spec['timestep_limit'] = min(gym_env.spec.timestep_limit,
                                     args.timestep_limit)
    env_render = env_render if args.render else None

    print '* environment', args.env
    print 'observation shape', env_spec['observation_shape']
    print 'action space', gym_env.action_space
    print 'timestep limit', env_spec['timestep_limit']
    print 'reward threshold', gym_env.spec.reward_threshold

    # model
    model = importlib.import_module('models.%s' % args.model)

    # train
    # gym monitor
    if args.monitor:
        env.monitor.start(args['monitor_dir'])

    train_q(env_spec, env_step, env_reset, env_render, vars(args),
          model.build_q_model)

    if args.monitor:
        env.monitor.close()
