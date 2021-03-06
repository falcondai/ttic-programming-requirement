import tensorflow as tf
import numpy as np

def build_model(observation_shape, n_actions, batch=None, n_cnn_layers=4, n_cnn_filters=32, n_fc_dim=256):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')
    tf.add_to_collection('inputs', obs_ph)

    # expect stacked frames, shape = [batch, frame, x, y, channel]
    # transform it into [batch, x, y, channel * frame]
    frames = tf.unstack(obs_ph, axis=1)
    obs = tf.concat(3, frames)

    net = obs / 255.
    for i in xrange(n_cnn_layers):
        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=n_cnn_filters,
            kernel_size=(3, 3),
            stride=(2, 2),
            activation_fn=tf.nn.elu,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv%i' % (i+1),
        )
    net = tf.contrib.layers.flatten(net)

    # feedforward
    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=n_fc_dim,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        scope='ff_fc1',
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
