import tensorflow as tf
import numpy as np

def build_q_model(observation_shape, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    net = obs_ph / 255.
    net = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=8,
        kernel_size=(7, 7),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv0',
    )
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    net = tf.contrib.layers.convolution2d(
        inputs=net,
        num_outputs=16,
        kernel_size=(9, 9),
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        scope='conv1',
    )
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    net = tf.contrib.layers.flatten(net)

    net = tf.contrib.layers.fully_connected(
        # inputs=tf.nn.dropout(net, keep_prob_ph),
        inputs=net,
        num_outputs=64,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        scope='fc0'
    )

    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=32,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        scope='fc1',
    )
    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=dim_action,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='fc2',
    )
    # tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

    action_values = net
    tf.add_to_collection('outputs', action_values)

    return obs_ph, keep_prob_ph, action_values
