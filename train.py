#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools, json
import tqdm
import argparse
import importlib
import gym
from gym import envs

def get_current_run_id(checkpoint_dir):
    paths = glob.glob('%s/hyperparameters.*.json' % checkpoint_dir)
    if len(paths) == 0:
        return 0
    return sorted(map(lambda p: int(p.split('.')[-2]), paths))[-1] + 1

def restore_vars(saver, sess, checkpoint_dir, restart=False):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
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

def wrapped_env(env, output_state, output_image):
    pass

def load_rollout(data_dir):
    pass

def rollout(behavior_policy, env, max_t, render_env=False, reset_env=True,
            last_obs=None):
    '''rollout based on behavior policy with options to continue from
    an existing environment'''
    if reset_env:
        obs = env.reset()
    else:
        obs = last_obs
    observations, actions, rewards = [], [], []
    done = False
    t = 0
    while not done and t < max_t:
        action_probs = behavior_policy(obs)
        action = np.random.choice(env.action_space.n, p=action_probs)
        observations.append(obs)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if render_env:
            env.render()
        t += 1
    return observations, actions, rewards, done

def vector_slice(A, B):
    """ Returns values of rows i of A at column B[i]

    where A is a 2D Tensor with shape [None, D]
    and B is a 1D Tensor with shape [None]
    with type int32 elements in [0,D)

    Example:
      A =[[1,2], B = [0,1], vector_slice(A,B) -> [1,4]
          [3,4]]
    """
    linear_index = (tf.shape(A)[1] * tf.range(0, tf.shape(A)[0]))
    linear_A = tf.reshape(A, [-1])
    return tf.gather(linear_A, B + linear_index)

def train(env, args, build_model):
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
        obs_ph, keep_prob_ph, logits, probs, state_value = build_model(
            env.observation_space.shape[0],
            env.action_space.n)
        actions_taken_ph = tf.placeholder('int32')
        avg_len_episode_ph = tf.placeholder('float')
        avg_episode_reward_ph = tf.placeholder('float')
        max_episode_reward_ph = tf.placeholder('float')
        min_episode_reward_ph = tf.placeholder('float')
        avg_tick_reward_ph = tf.placeholder('float')

        # expected reward under policy
        # entropy regularizer to encourage action diversity
        entropy_reg = - tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs), 1))
        action_logits = vector_slice(tf.log(probs), actions_taken_ph)
        observed_reward_ph = tf.placeholder('float')
        rewards_to_go_ph = tf.placeholder('float')

        # with total episodic reward
        # objective = observed_reward_ph * tf.reduce_sum(action_logits) \
        #     + args['reg_coeff'] * entropy_reg

        # with rewards to go
        objective = tf.reduce_sum(action_logits * rewards_to_go_ph) \
            + args['reg_coeff'] * entropy_reg

        # optimization
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(
            args['initial_learning_rate'],
            global_step, args['n_decay_steps'],
            args['decay_rate'],
            staircase=not args['no_decay_staircase'])

        if args['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate,
                                               args['adam_beta1'],
                                               args['adam_beta2'],
                                               args['adam_epsilon'],
                                              )
        elif args['optimizer'] == 'ag':
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   args['momentum'],
                                                   use_nesterov=True,
                                                  )
        elif args['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                   args['rmsprop_decay'],
                                                   args['momentum'],
                                                   args['rmsprop_epsilon'],
                                                  )
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   args['momentum'],
                                                  )

        # train ops
        grad_vars = optimizer.compute_gradients(-objective)
        update_policy_op = optimizer.apply_gradients(
            grad_vars,
            global_step=global_step,
        )

        # summary
        if not args['no_summary']:
            tf.scalar_summary('learning_rate', learning_rate)
            tf.scalar_summary('average_episode_reward', avg_episode_reward_ph)
            tf.scalar_summary('max_episode_reward', max_episode_reward_ph)
            tf.scalar_summary('min_episode_reward', min_episode_reward_ph)
            tf.scalar_summary('average_tick_reward', avg_tick_reward_ph)
            tf.scalar_summary('average_episode_length', avg_len_episode_ph)
            # tf.scalar_summary('regularizer', entropy_reg)
            # tf.scalar_summary('objective', objective)

            print '* extra summary'
            for g, v in grad_vars:
                tf.histogram_summary('gradients/%s' % v.name, g)
                print 'gradients/%s' % v.name

            summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
        with tf.Session() as sess:
            if not args['no_summary']:
                writer = tf.train.SummaryWriter(summary_dir, sess.graph,
                                                flush_secs=30)
                print '* writing summary to', summary_dir
            restore_vars(saver, sess, args['checkpoint_dir'], args['restart'])

            print '* regularized parameters:'
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                print v.name

            env_spec = envs.registry.env_specs[args['env']]
            print '* environment', args['env']
            print 'observation space', env.observation_space
            print 'action space', env.action_space
            print 'timestep limit', env_spec.timestep_limit
            print 'reward threshold', env_spec.reward_threshold

            # stochastic policy
            policy = lambda obs: probs.eval(feed_dict={
                obs_ph: [obs],
                keep_prob_ph: 1. - args['dropout_rate'],
            })[0]

            # gym monitor
            if args['monitor']:
                env.monitor.start(args['monitor_dir'])

            for i in tqdm.tqdm(xrange(args['n_train_steps'])):
                # on-policy rollout for some episodes
                episodes = []
                n_ticks = 0
                episode_rewards = []
                for j in xrange(args['n_update_episodes']):
                    observations, actions, rewards, done = rollout(
                        policy,
                        env,
                        env_spec.timestep_limit,
                        args['render_env'],
                        reset_env=True,
                    )
                    episodes.append((observations, actions, rewards))
                    n_ticks += len(observations)
                    episode_rewards.append(np.sum(rewards))

                avg_len_episode = n_ticks * 1. / args['n_update_episodes']

                # transform and preprocess the rollouts
                obs = []
                action_inds = []
                # the objective function values
                f_vals = []
                for observations, actions, rewards in episodes:
                    len_episode = len(observations)
                    obs += observations
                    action_inds += actions
                    # total episodic reward with lambda decay over ticks
                    # f_vals += [np.sum(np.prod([
                    #     rewards,
                    #     [args['reward_lambda']**t for t in xrange(len_episode)]],
                    #     axis=0))
                    # ] * len_episode

                    # rewards to go with lambda decay
                    f_vals = [np.sum(np.prod([
                        rewards[t:],
                        [args['reward_lambda']**u for u in xrange(len_episode-t)]],
                        axis=0))
                    for t in xrange(len_episode)]

                # estimate policy gradient by batches
                # accumulate gradients over batches
                acc_grads = dict([(grad, np.zeros(grad.get_shape()))
                                      for grad, var in grad_vars])
                n_batch = int(np.ceil(n_ticks * 1. / args['n_batch_ticks']))
                for j in xrange(n_batch):
                    start = j * args['n_batch_ticks']
                    end = min(start + args['n_batch_ticks'], n_ticks)
                    grad_feed = {
                        obs_ph: obs[start:end],
                        keep_prob_ph: 1. - args['dropout_rate'],
                        actions_taken_ph: action_inds[start:end],
                        # observed_reward_ph: episode_rewards[start:end],
                        rewards_to_go_ph: f_vals[start:end],
                    }

                    # compute the expectation of gradients
                    grad_vars_val = sess.run([
                        grad_vars,
                        ], feed_dict=grad_feed)[0]
                    for (g, _), (g_val, _) in zip(grad_vars, grad_vars_val):
                        acc_grads[g] += g_val / args['n_update_episodes']

                # update policy with the sample expectation of gradients
                update_dict = {
                    avg_len_episode_ph: avg_len_episode,
                    avg_episode_reward_ph: np.mean(episode_rewards),
                    max_episode_reward_ph: np.max(episode_rewards),
                    min_episode_reward_ph: np.min(episode_rewards),
                    avg_tick_reward_ph: np.sum(episode_rewards) * 1. / n_ticks,
                }
                update_dict.update(acc_grads)
                summary_val, _ = sess.run([summary_op, update_policy_op],
                                          feed_dict=update_dict)

                if not args['no_summary']:
                    writer.add_summary(summary_val, global_step.eval())

                if i % args['n_save_interval'] == 0:
                    saver.save(sess, args['checkpoint_dir'] + '/model',
                               global_step=global_step.eval())

            # save again at the end
            saver.save(sess, args['checkpoint_dir'] + '/model',
                       global_step=global_step.eval())

            if args['monitor']:
                env.monitor.close()

def build_argparser():
    parse = argparse.ArgumentParser()

    # gym options
    parse.add_argument('--env', default='CartPole-v0')
    parse.add_argument('--model', required=True)
    parse.add_argument('--monitor', action='store_true')
    parse.add_argument('--monitor_dir',
                       default='/tmp/gym-monitor-%i' % time.time())

    parse.add_argument('--reg_coeff', type=float, default=0.0001)
    parse.add_argument('--reward_lambda', type=float, default=1.)
    parse.add_argument('--dropout_rate', type=float, default=0.2)

    parse.add_argument('--restart', action='store_true')
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--no_summary', action='store_true')
    parse.add_argument('--summary_prefix', default='')
    parse.add_argument('--render_env', action='store_true')

    # how many episodes to rollout before update parameters
    parse.add_argument('--n_update_episodes', type=int, default=4)
    parse.add_argument('--n_batch_ticks', type=int, default=128)
    parse.add_argument('--n_save_interval', type=int, default=128)
    parse.add_argument('--n_train_steps', type=int, default=1e5)

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
    parse.add_argument('--n_decay_steps', type=int, default=512)
    parse.add_argument('--no_decay_staircase', action='store_true')
    parse.add_argument('--decay_rate', type=float, default=0.8)

    parse.add_argument('--np_seed', type=int, default=123)
    parse.add_argument('--tf_seed', type=int, default=1234)

    return parse


if __name__ == '__main__':
    from functools import partial

    # arguments
    parse = build_argparser()
    args = parse.parse_args()

    env = gym.make(args.env)

    # model
    model = importlib.import_module('models.%s' % args.model)

    # train
    train(env, vars(args), model.build_model)
