import tensorflow as tf
import numpy as np

class CnnCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, observation_shape, n_actions, n_cnn_layers, n_rnn_dim):
        self.n_actions = n_actions
        self.n_cnn_layers = n_cnn_layers
        self.n_rnn_dim = n_rnn_dim
        self.observation_shape = list(observation_shape)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or 'CnnCell'):
            # conv part
            inputs = tf.reshape(inputs, [1] + self.observation_shape)
            net = inputs / 255.
            for i in xrange(self.n_cnn_layers):
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

            # rnn
            self.rnn = tf.nn.rnn_cell.BasicLSTMCell(self.n_rnn_dim, state_is_tuple=False)
            rnn_output, rnn_state = self.rnn(net, state)

            # prediction outputs
            action_logits = tf.contrib.layers.fully_connected(
                inputs=rnn_output,
                num_outputs=self.n_actions,
                biases_initializer=tf.zeros_initializer,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None,
                scope='action_fc1',
            )
            action_probs = tf.nn.softmax(action_logits, name='action_probs')

            state_value = tf.contrib.layers.fully_connected(
                inputs=rnn_output,
                num_outputs=1,
                biases_initializer=tf.zeros_initializer,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None,
                scope='value_fc1',
            )

            return (action_probs, state_value), rnn_state

    @property
    def state_size(self):
        return self.n_rnn_dim

    @property
    def output_size(self):
        return (self.n_actions, 1)


def build_model(observation_shape, n_actions, batch=None, n_cnn_layers=4, n_rnn_dim=256):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape),
                            name='observation')
    seq_len_ph = tf.placeholder('int32', name='sequence_lengths')
    initial_state_ph = tf.placeholder('float', [batch, 2*n_rnn_dim], name='initial_state')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', seq_len_ph)
    tf.add_to_collection('inputs', initial_state_ph)

    input_size = np.prod(observation_shape)
    flattened_obs = tf.reshape(obs_ph, [1, -1, input_size])
    cell = CnnCell(observation_shape, n_actions, n_cnn_layers, n_rnn_dim)
    # TODO try time-major without flattening inputs
    (action_probs, state_values), final_state = tf.nn.dynamic_rnn(cell, flattened_obs,
                                                                  seq_len_ph, initial_state_ph, parallel_iterations=1)

    # flatten the batch dim so the time dim becomes batch dim for downstream
    # learning with policy gradients
    action_probs = tf.squeeze(action_probs, 0)
    state_values = tf.squeeze(state_values, 0)

    tf.add_to_collection('outputs', action_probs)
    tf.add_to_collection('outputs', state_values)

    # policy function
    def pi_func(obs_val, rnn_state_val):
        sess = tf.get_default_session()
        p, next_rnn_state_val = sess.run([action_probs, final_state], {
            obs_ph: [obs_val],
            initial_state_ph: [rnn_state_val],
            seq_len_ph: [1],
        })
        return np.random.choice(n_actions, p=p[0]), next_rnn_state_val[0]

    # value function
    def v_func(obs_val, rnn_state_val):
        return state_values.eval(feed_dict={
            obs_ph: [obs_val],
            initial_state_ph: [rnn_state_val],
            seq_len_ph: [1],
        })[0, 0]

    return obs_ph, seq_len_ph, initial_state_ph, action_probs, state_values, final_state, \
    pi_func, v_func
