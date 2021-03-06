#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, time, importlib, argparse, itertools
from envs import get_env
from agents import get_agent_builder, StatefulAgent

def evaluate(env_spec, env_step, env_reset, env_render, agent, n_episodes=None):
    # evaluation
    episode_rewards = []
    episode_lengths = []

    is_stateful = isinstance(agent, StatefulAgent)
    for i in xrange(n_episodes) if n_episodes else itertools.count():
        ob = env_reset()
        if env_render != None:
            env_render()
        done = False
        if is_stateful:
            agent.reset()
        rewards = []
        while not done:
            action = agent.act(ob)
            ob, reward, done = env_step(action)
            rewards.append(reward)
            if env_render != None:
                env_render()
        episode_rewards.append(np.sum(rewards))
        episode_lengths.append(len(rewards))
        print 'episode', i, 'reward', episode_rewards[-1], 'length', episode_lengths[-1]

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
    parser.add_argument('-n', '--n-episodes', type=int, default=0)
    parser.add_argument('--render-fps', type=float, default=60.)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('-e', '--env-id', default='atari.skip.quarter.Pong')
    parser.add_argument('-a', '--agent', required=True)
    parser.add_argument('-u', '--unwrap', type=int, default=0)

    args = parser.parse_args()

    if not args.checkpoint_path:
        # use the latest checkpoint in the folder
        checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint_path = args.checkpoint_path

    if checkpoint_path is None:
        print '* checkpoint path does not exist'
    else:
        # init env
        env = get_env(args.env_id)
        env.render_fps = args.render_fps
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
        with tf.variable_scope('global'):
            agent = get_agent_builder(args.agent)(env.spec)
        saver = tf.train.Saver()

        # eval
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            print 'restored checkpoint from %s' % checkpoint_path
            evaluate(env.spec, env.step, env.reset, env_render, agent, None if args.n_episodes==0 else args.n_episodes)
