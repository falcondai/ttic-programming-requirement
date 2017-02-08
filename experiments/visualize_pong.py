#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import importlib, argparse
from util import partial_rollout
from envs import get_env
from visualize import HorizonChart, ImageStackChart
from test import evaluate

def wrap_policy(n_cnn_layers=4):
    obs_ph, initial_state_ph = tf.get_collection('inputs', scope='global')
    action_logits, state_values, final_state = tf.get_collection('outputs', scope='global')

    action_probs = tf.nn.softmax(action_logits)
    log_action_probs = tf.nn.log_softmax(action_logits)
    action_entropy = -tf.reduce_sum(log_action_probs * action_probs)
    perplexity = tf.exp(action_entropy)
    action = tf.multinomial(action_logits, 1)

    # visualize intermediate variables
    g = tf.get_default_graph()
    convs = [g.get_tensor_by_name('global/conv%i/Elu:0' % (k+1)) for k in xrange(n_cnn_layers)]

    # charts
    value_chart = HorizonChart(1000, 0.005, 100, title='state value')
    perplexity_chart = HorizonChart(1000, 6./100, 100, title='action perplexity')
    conv_charts = [ImageStackChart(2.**k, title='conv%i' % (k+1)) for k in xrange(n_cnn_layers)]

    def pi_v_func(obs_val, history):
        sess = tf.get_default_session()

        fetched = sess.run(convs + [action, state_values, perplexity, final_state], {
            obs_ph: [obs_val],
            initial_state_ph: history,
        })
        conv_val = fetched[:n_cnn_layers]
        action_val, state_value_val, perplexity_val, final_state_val = fetched[n_cnn_layers:]

        # update and draw charts
        for k, (chart, conv) in enumerate(zip(conv_charts, conv_val)):
            chart.update(conv[0])
            chart.draw()

        value_chart.update(state_value_val)
        value_chart.draw()

        perplexity_chart.update(perplexity_val)
        perplexity_chart.draw()

        return action_val[0, 0], state_value_val[0], final_state_val

    return pi_v_func

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--checkpoint-dir', default='experiments/pong-zero-checkpoints')
    parser.add_argument('--checkpoint-path')
    parser.add_argument('-e', '--env-id', default='atari.skip.quarter.Pong')
    parser.add_argument('-m', '--model', default='cnn_gru_pi_v')

    args = parser.parse_args()

    if not args.checkpoint_path:
        # use the latest checkpoint in the folder
        checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint_path = args.checkpoint_path

    # init env
    env = get_env(args.env_id)
    # HACK use the base env's render
    env_render = env.env.env.env.render

    print '* environment'
    print env.spec

    # build model
    model = importlib.import_module('models.%s' % args.model)
    with tf.variable_scope('global'):
        pi, v, z = model.build_model(env.spec['observation_shape'], env.spec['action_size'])[-3:]
        pi = wrap_policy()
    saver = tf.train.Saver()

    # eval
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        print 'restored checkpoint from %s' % checkpoint_path
        evaluate(env.spec, env.step, env.reset, env_render, pi, z)
