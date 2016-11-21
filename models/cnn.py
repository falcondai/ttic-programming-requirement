import tensorflow as tf
import numpy as np

def build_model(observation_shape, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    with tf.variable_scope('model'):
        net = obs_ph / 255.
        for i in xrange(3):
            net = tf.contrib.layers.convolution2d(
                inputs=net,
                num_outputs=32,
                kernel_size=(5, 5),
                activation_fn=tf.nn.relu,
                biases_initializer=tf.zeros_initializer,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                scope='conv%i' % i,
            )
            # tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

        net = tf.contrib.layers.flatten(net)
        for i in xrange(1):
            net = tf.contrib.layers.fully_connected(
                # inputs=tf.nn.dropout(net, keep_prob_ph),
                inputs=net,
                num_outputs=32,
                biases_initializer=tf.zeros_initializer,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu,
                scope='fc%i' % i,
            )
            # tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

        net = tf.contrib.layers.fully_connected(
            # inputs=tf.nn.dropout(obs_ph, keep_prob_ph),
            inputs=net,
            num_outputs=dim_action,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None,
            scope='fc%i' % (i + 1),
        )
        # tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

        logits = net
        probs = tf.nn.softmax(logits, name='probs')
        tf.add_to_collection('outputs', logits)
        tf.add_to_collection('outputs', probs)

    return obs_ph, keep_prob_ph, logits, probs, None
