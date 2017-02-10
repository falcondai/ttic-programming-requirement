#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import importlib, argparse
from util import partial_rollout
from envs import get_env
from visualize import HorizonChart, ImageStackChart
from test import evaluate

def wrap_policy(obs_ph, initial_state_ph, action_logits, state_values, final_state, n_cnn_layers=4, value_resolution=0.005, action_resolution=6./100):
    action_probs = tf.nn.softmax(action_logits)
    log_action_probs = tf.nn.log_softmax(action_logits)
    action_entropy = -tf.reduce_sum(log_action_probs * action_probs)
    perplexity = tf.exp(action_entropy)
    action = tf.multinomial(action_logits, 1)

    # visualize intermediate variables
    g = tf.get_default_graph()
    convs = [g.get_tensor_by_name('global/conv%i/Elu:0' % (k+1)) for k in xrange(n_cnn_layers)]

    # charts
    value_chart = HorizonChart(1000, value_resolution, 100, title='state value')
    perplexity_chart = HorizonChart(1000, action_resolution, 100, title='action perplexity')
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
    parser.add_argument('-u', '--unwrap', type=int, default=3)
    parser.add_argument('--n-cnn-layers', type=int, default=0)
    parser.add_argument('--value-resolution', type=float, default=0.005)

    args = parser.parse_args()

    if not args.checkpoint_path:
        # use the latest checkpoint in the folder
        checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint_path = args.checkpoint_path

    # init env
    env = get_env(args.env_id)
    # HACK unwrap to use base env's render
    _env = env
    for i in xrange(args.unwrap):
        _env = _env.env
    env_render = _env.render

    print '* environment'
    print env.spec

    # build model
    model = importlib.import_module('models.%s' % args.model)
    with tf.variable_scope('global'):
        model_vs = model.build_model(env.spec['observation_shape'], env.spec['action_size'])
        pi, v, z = model_vs[-3:]
        pi = wrap_policy(*model_vs[:5], n_cnn_layers=args.n_cnn_layers, value_resolution=args.value_resolution)
    saver = tf.train.Saver()

    # eval
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        print 'restored checkpoint from %s' % checkpoint_path
        evaluate(env.spec, env.step, env.reset, env_render, pi, z)
