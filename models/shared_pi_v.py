import tensorflow as tf
import numpy as np

def build_model(observation_shape, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    net = tf.contrib.layers.fully_connected(
        inputs=obs_ph,
        num_outputs=16,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        scope='fc1'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

    # policy logits
    action_logits = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=dim_action,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='action_fc2'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, action_logits)

    # state-value
    val = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=1,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='val_fc1'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, val)

    action_probs = tf.nn.softmax(action_logits, name='action_probs')
    state_value = val
    tf.add_to_collection('outputs', action_probs)
    tf.add_to_collection('outputs', state_value)

    return obs_ph, keep_prob_ph, action_probs, state_value
