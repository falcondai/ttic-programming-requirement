import tensorflow as tf
import numpy as np

def build_q_model(observation_shape, dim_action, trainable=True,
                  batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    net = obs_ph
    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=32,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        trainable=trainable,
        scope='fc1'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)
    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=16,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        trainable=trainable,
        scope='fc2'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)
    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=dim_action,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        trainable=trainable,
        scope='fc3'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

    action_values = net
    tf.add_to_collection('outputs', action_values)

    return obs_ph, keep_prob_ph, action_values
