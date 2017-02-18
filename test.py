#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, time, importlib, argparse, itertools
from util import partial_rollout
from envs import get_env
from agents.adv_ac import A3CAgent

def evaluate(env_spec, env_step, env_reset, env_render, policy, zero_state, n_episodes=None):
    # evaluation
    episode_rewards = []
    episode_lengths = []

    ro_gen = partial_rollout(env_reset, env_step, policy, zero_state, None, env_render)
    for i in xrange(n_episodes) if n_episodes else itertools.count():
        ro = ro_gen.next()
        rewards = ro[2]
        episode_rewards.append(np.sum(rewards))
        episode_lengths.append(len(rewards))
        print 'episode', i, 'reward', episode_rewards[-1], 'length', episode_lengths[-1]

    # summary
    print '* summary'
    print 'episode lengths:',
    print 'mean', np.mean(episode_lengths), '\t',
    print 'median', np.median(episode_lengths), '\t',
    print 'max', np.max(episode_lengths), '\t',
    print 'min', np.min(episode_lengths), '\t',
    print 'std', np.std(episode_lengths)

    print 'episode rewards:',
    print 'mean', np.mean(episode_rewards), '\t',
    print 'median', np.median(episode_rewards), '\t',
    print 'max', np.max(episode_rewards), '\t',
    print 'min', np.min(episode_rewards), '\t',
    print 'std', np.std(episode_rewards)

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--checkpoint-dir')
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--n-episodes', type=int, default=0)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('-e', '--env-id', default='atari.skip.quarter.Pong')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-u', '--unwrap', type=int, default=0)

    args = parser.parse_args()

    if not args.checkpoint_path:
        # use the latest checkpoint in the folder
        checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint_path = args.checkpoint_path

    # init env
    env = get_env(args.env_id)
    if not args.no_render:
        # HACK unwrap to use base env's render
        _env = env
        for i in xrange(args.unwrap):
            _env = _env.env
        env_render = _env.render
    else:
        env_render = None

    print '* environment'
    print env.spec

    # build model
    model = importlib.import_module('models.%s' % args.model)
    with tf.variable_scope('global'):
        agent = A3CAgent(env.spec, model.build_model)
    saver = tf.train.Saver()

    # eval
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        print 'restored checkpoint from %s' % checkpoint_path
        zero_state = agent.zero_state if agent.spec['use_history'] else None
        evaluate(env.spec, env.step, env.reset, env_render, agent.act, zero_state, None if args.n_episodes==0 else args.n_episodes)
