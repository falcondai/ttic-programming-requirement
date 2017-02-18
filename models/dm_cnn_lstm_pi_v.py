# the model definition is based on the one LSTM RNN used in DeepMind A3C paper
# arXiv:1602.01783v2

import tensorflow as tf
import numpy as np

def build_model(observation_shape, n_actions, batch=None, n_rnn_dim=256):
    assert len(observation_shape) == 3

    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    initial_state_ph = tf.placeholder('float', [2, batch, n_rnn_dim], name='initial_state')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', initial_state_ph)

    net = obs_ph / 255.
    net = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=16,
        kernel_size=(8, 8),
        stride=(4, 4),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv1',
    )
    net = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=32,
        kernel_size=(4, 4),
        stride=(2, 2),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv2',
    )

    net = tf.contrib.layers.flatten(net)
    rnn_input = tf.expand_dims(net, 0)

    # rnn
    cell = tf.contrib.rnn.LSTMBlockCell(n_rnn_dim)

    lstm_state_tuple = tuple(tf.unstack(initial_state_ph))
    seq_len = tf.shape(obs_ph)[:1]
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_len, lstm_state_tuple)
    rnn_outputs = tf.squeeze(rnn_outputs, 0)

    # prediction outputs
    action_logits = tf.contrib.layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=n_actions,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='action_fc1',
    )

    state_values = tf.contrib.layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=1,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='value_fc1',
    )
    state_values = tf.squeeze(state_values, -1)

    tf.add_to_collection('outputs', action_logits)
    tf.add_to_collection('outputs', state_values)
    tf.add_to_collection('outputs', final_state)

    zero_state = np.zeros((2, 1, n_rnn_dim))

    return obs_ph, initial_state_ph, \
    action_logits, state_values, \
    final_state, zero_state
