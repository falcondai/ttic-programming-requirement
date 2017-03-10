import tensorflow as tf
import numpy as np

def build_model(observation_shape, n_actions, batch=None, n_fc_layers=1, n_fc_dim=32):
    assert len(observation_shape) == 1

    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')

    tf.add_to_collection('inputs', obs_ph)

    net = obs_ph
    for i in xrange(n_fc_layers):
        net = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=n_fc_dim,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=tf.nn.elu,
            scope='fc%i' % (i + 1),
        )

    # prediction outputs
    action_logits = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=n_actions,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='action_fc1',
    )

    state_values = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=1,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='value_fc1',
    )
    state_values = tf.squeeze(state_values, -1)

    tf.add_to_collection('outputs', action_logits)
    tf.add_to_collection('outputs', state_values)

    return obs_ph, \
    action_logits, state_values
