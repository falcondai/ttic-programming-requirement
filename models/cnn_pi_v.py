import tensorflow as tf
import numpy as np

def build_model(observation_shape, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    net = obs_ph / 255.
    for i in xrange(4):
        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            activation_fn=tf.nn.elu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv%i' % (i+1),
        )

    net = tf.contrib.layers.flatten(net)

    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=64,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.elu,
        scope='fc0',
    )

    action_logits = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=dim_action,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='a_fc1',
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, action_logits)

    state_value = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=1,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='v_fc1',
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, state_value)

    probs = tf.nn.softmax(action_logits, name='probs')
    tf.add_to_collection('outputs', probs)
    tf.add_to_collection('outputs', state_value)

    return obs_ph, keep_prob_ph, probs, state_value
