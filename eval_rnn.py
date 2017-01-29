#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools, csv, glob
from functools import partial
import tqdm
import argparse
import gym
from gym import envs
from util import rollout, vector_slice, to_greedy, to_epsilon_greedy, \
                 test_restore_vars, atari_env
from train_unified import partial_rollout
import cv2
from visualize import HorizonChart, ImageStackChart

value_chart = HorizonChart(1000, 0.005, 100)
perplexity_chart = HorizonChart(1000, 6./100, 100)
conv_chart = ImageStackChart()

def visualize(win_name, conv_output, fps=30.):
    im = np.transpose(conv_output[0], [2, 0, 1])
    # normalize per image
    min_im = im.min(1, keepdims=True).min(2, keepdims=True)
    ptp_im = im.ptp(1).ptp(1).reshape((-1, 1, 1))
    im = (im - min_im) / ptp_im

    im = np.hstack(im)
    cv2.imshow(win_name, im)
    cv2.waitKey(int(1000. / fps))
    # pause at each frame, need to manually unpause
    # cv2.waitKey(0)

def load_policy(sess, checkpoint_path, meta_path, model_type, policy_type, epsilon):
    test_restore_vars(sess, checkpoint_path, meta_path)

    obs_ph, initial_state_ph = tf.get_collection('inputs')
    action_logits, state_values, final_state = tf.get_collection('outputs')

    action_probs = tf.nn.softmax(action_logits)
    log_action_probs = tf.nn.log_softmax(action_logits)
    action_entropy = -tf.reduce_sum(log_action_probs * action_probs)
    perplexity = tf.exp(action_entropy)
    action = tf.multinomial(action_logits, 1)

    # visualize intermediate variables
    g = tf.get_default_graph()
    conv1 = g.get_tensor_by_name('model/conv1/Elu:0')
    conv2 = g.get_tensor_by_name('model/conv2/Elu:0')
    conv3 = g.get_tensor_by_name('model/conv3/Elu:0')
    conv4 = g.get_tensor_by_name('model/conv4/Elu:0')
    def pi_v_func(obs_val, rnn_state_val):
        sess = tf.get_default_session()

        conv1_val, conv2_val, conv3_val, conv4_val, action_val, state_value_val, next_rnn_state_val, perplexity_val = sess.run([conv1, conv2, conv3, conv4, action, state_values, final_state, perplexity], {
            obs_ph: [obs_val],
            initial_state_ph: rnn_state_val,
        })
        # visualize('conv1', conv1_val)
        # visualize('conv2', conv2_val)
        # visualize('conv3', conv3_val)
        # visualize('conv4', conv4_val)

        # conv_chart.update(conv1_val[0])
        # cv2.imshow('conv1', conv_chart.im)
        # cv2.waitKey(int(1000./30.))

        value_chart.update(state_value_val)
        cv2.imshow('value', value_chart.im)
        cv2.waitKey(int(1000./30.))
        perplexity_chart.update(perplexity_val)
        cv2.imshow('action perplexity', perplexity_chart.im)
        cv2.waitKey(int(1000./30.))

        # disturb
        if np.random.rand() < 0.1:
            action_val[0, 0] = np.random.randint(6)

        return action_val[0, 0], state_value_val[0], next_rnn_state_val

    return pi_v_func

def evaluate(env_spec, env_step, env_reset,
             env_render, n_samples, policy):
    # evaluation
    episode_rewards = []
    episode_lengths = []

    for ro in partial_rollout((env_spec, env_step, env_reset, env_render), policy, zero_state=np.zeros((1, 512)), n_ticks=None):
        rewards = ro[2]
        episode_rewards.append(np.sum(rewards))
        episode_lengths.append(len(rewards))

    # summary
    print '* summary'
    print 'episode lengths:',
    print 'mean', np.mean(episode_lengths),
    print 'median', np.median(episode_lengths),
    print 'max', np.max(episode_lengths),
    print 'min', np.min(episode_lengths),
    print 'std', np.std(episode_lengths)

    print 'episode rewards:',
    print 'mean', np.mean(episode_rewards),
    print 'median', np.median(episode_rewards),
    print 'max', np.max(episode_rewards),
    print 'min', np.min(episode_rewards),
    print 'std', np.std(episode_rewards)

if __name__ == '__main__':
    from util import passthrough, use_render_state
    from envs import GymEnv, GrayscaleWrapper, ScaleWrapper

    # arguments
    parse = argparse.ArgumentParser()
    parse.add_argument('--checkpoint_path', required=True)
    parse.add_argument('--meta_path')
    parse.add_argument('--latest', action='store_true')
    parse.add_argument('--no_render', action='store_true')
    parse.add_argument('--n_samples', type=int, default=16)
    parse.add_argument('--env', default='PongDeterministic-v3')
    parse.add_argument('--model', choices=['q', 'pi', 'pi_v'], default='pi')
    parse.add_argument('--policy', choices=['greedy', 'epsilon_greedy',
                                            'sample'], default='sample')
    parse.add_argument('--epsilon', type=float, default=1e-5)
    parse.add_argument('--n_obs_ticks', type=int, default=1)
    parse.add_argument('--timestep_limit', type=int, default=10**9)
    parse.add_argument('--use_render_state', action='store_true')
    parse.add_argument('--scale', type=float, default=0.27)
    parse.add_argument('--interpolation', choices=['nearest', 'bilinear',
                                                   'bicubic', 'cubic'],
                       default='nearest')

    args = parse.parse_args()

    if args.latest:
        # use the latest checkpoint in the folder
        checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)
    else:
        checkpoint_path = args.checkpoint_path

    if args.meta_path == None:
        # find a meta graph file in the same directory
        meta_path = glob.glob(os.path.dirname(checkpoint_path) + '/*.meta')[0]
    else:
        meta_path = args.meta_path

    # init env
    gym_env = gym.make(args.env)
    env_spec, env_step, env_reset, env_render = passthrough(gym_env)
    env_spec, env_step, env_reset, env_render = atari_env((env_spec, env_step, env_reset, env_render), 43./160, 1)
    # gym_env = GymEnv(args.env)
    # env = GrayscaleWrapper(ScaleWrapper(gym_env, 42./160.))
    # env_render = None if args.no_render else env.render

    # print '* environment', args.env
    # print 'observation shape', env.spec['observation_shape']
    # print 'action space', env.spec['action_size']
    # print 'timestep limit', env.spec['timestep_limit']
    # print 'reward threshold', gym_env.spec.reward_threshold

    # eval
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            policy = load_policy(sess, checkpoint_path, meta_path, args.model, args.policy, args.epsilon)
            # evaluate(env.spec, env.step, env.reset, env_render, args.n_samples, policy)
            evaluate(env_spec, env_step, env_reset, env_render, args.n_samples, policy)
