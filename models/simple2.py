import tensorflow as tf
import numpy as np

def build_model(dim_observation, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch, dim_observation], name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    with tf.variable_scope('model'):
        fc1 = tf.contrib.layers.fully_connected(
            # inputs=tf.nn.dropout(obs_ph, keep_prob_ph),
            inputs = obs_ph,
            num_outputs=64,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=tf.nn.sigmoid,
            scope='fc1'
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, fc1)

        fc2 = tf.contrib.layers.fully_connected(
            # inputs=tf.nn.dropout(obs_ph, keep_prob_ph),
            inputs = fc1,
            num_outputs=dim_action,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None,
            scope='fc2'
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, fc2)

        # val_fc1 = tf.contrib.layers.fully_connected(
        #     # inputs=tf.nn.dropout(obs_ph, keep_prob_ph),
        #     inputs = obs_ph,
        #     num_outputs=1,
        #     biases_initializer=tf.zeros_initializer,
        #     weights_initializer=tf.contrib.layers.xavier_initializer(),
        #     activation_fn=None,
        #     scope='val_fc1'
        # )
        # tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, val_fc1)

        logits = fc2
        probs = tf.nn.softmax(logits, name='probs')
        # value = val_fc1
        value = None
        tf.add_to_collection('outputs', logits)
        tf.add_to_collection('outputs', probs)
        # tf.add_to_collection('outputs', value)

    return obs_ph, keep_prob_ph, logits, probs, value
