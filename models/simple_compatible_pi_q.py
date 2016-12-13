import tensorflow as tf
import numpy as np

def build_model(observation_shape, dim_action, batch=None):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape),
                            name='observation')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    # policy logits
    net = tf.contrib.layers.fully_connected(
        inputs=obs_ph,
        num_outputs=dim_action,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='fc1'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, net)

    # state-value
    # TODO compute from features of (s, a)
    val_fc1 = tf.contrib.layers.fully_connected(
        # inputs=tf.gradients(tf.log(action_probs), ),
        num_outputs=1,
        biases_initializer=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='val_fc1'
    )
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, val_fc1)

    action_probs = tf.nn.softmax(net, name='action_probs')
    value = val_fc1
    tf.add_to_collection('outputs', action_probs)
    tf.add_to_collection('outputs', value)

    return obs_ph, keep_prob_ph, action_probs, value
