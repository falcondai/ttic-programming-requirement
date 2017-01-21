import tensorflow as tf
import numpy as np

class FcCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, observation_shape, n_actions, n_rnn_dim):
        self.n_actions = n_actions
        self.n_rnn_dim = n_rnn_dim
        self.observation_shape = list(observation_shape)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or 'FcCell'):
            # conv part
            inputs = tf.reshape(inputs, [1] + self.observation_shape)
            net = inputs
            net = tf.contrib.layers.fully_connected(
                inputs=net,
                num_outputs=16,
                activation_fn=tf.nn.elu,
                biases_initializer=tf.zeros_initializer,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                scope='fc1',
            )

            # rnn
            self.rnn = tf.contrib.rnn.GRUBlockCell(self.n_rnn_dim)
            # self.rnn = tf.nn.rnn_cell.BasicRNNCell(self.n_rnn_dim)
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


def build_model(observation_shape, n_actions, batch=None, n_rnn_dim=16):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape),
                            name='observation')
    seq_len_ph = tf.placeholder('int32', name='sequence_lengths')
    initial_state_ph = tf.placeholder('float', [batch, n_rnn_dim], name='initial_state')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', seq_len_ph)
    tf.add_to_collection('inputs', initial_state_ph)

    input_size = np.prod(observation_shape)
    flattened_obs = tf.reshape(obs_ph, [1, -1, input_size])
    cell = FcCell(observation_shape, n_actions, n_rnn_dim)
    seq_len = tf.shape(obs_ph)[:1]
    # TODO try time-major without flattening inputs
    (action_probs, state_values), final_state = tf.nn.dynamic_rnn(cell, flattened_obs,
                                                                  sequence_length=seq_len, initial_state=initial_state_ph, time_major=False)

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
