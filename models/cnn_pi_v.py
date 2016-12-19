import tensorflow as tf
import numpy as np

def build_model(observation_shape, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    with tf.variable_scope('model'):
        net = obs_ph / 255.
        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=16,
            kernel_size=(3, 3),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv0',
        )
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=16,
            kernel_size=(3, 3),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv1',
        )
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=16,
            kernel_size=(3, 3),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv2',
        )
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=16,
            kernel_size=(3, 3),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv3',
        )
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        net = tf.contrib.layers.flatten(net)

        # net = tf.contrib.layers.fully_connected(
        #     # inputs=tf.nn.dropout(net, keep_prob_ph),
        #     inputs=net,
        #     num_outputs=32,
        #     biases_initializer=tf.zeros_initializer,
        #     weights_initializer=tf.contrib.layers.xavier_initializer(),
        #     activation_fn=tf.nn.relu,
        #     scope='fc0'
        # )

        action_logits = tf.contrib.layers.fully_connected(
            # inputs=tf.nn.dropout(obs_ph, keep_prob_ph),
            inputs=net,
            num_outputs=dim_action,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None,
            scope='a_fc1',
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, action_logits)

        state_value = tf.contrib.layers.fully_connected(
            # inputs=tf.nn.dropout(obs_ph, keep_prob_ph),
            inputs=net,
            num_outputs=1,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None,
            scope='v_fc1',
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, action_logits)

        probs = tf.nn.softmax(action_logits, name='probs')
        tf.add_to_collection('outputs', probs)
        tf.add_to_collection('outputs', state_value)

    return obs_ph, keep_prob_ph, probs, state_value
