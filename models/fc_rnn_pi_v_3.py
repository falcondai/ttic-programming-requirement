import tensorflow as tf
import numpy as np

def build_model(observation_shape, n_actions, batch=None, n_rnn_dim=16):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape),
                            name='observation')
    seq_len_ph = tf.placeholder('int32', name='sequence_lengths')
    initial_state_ph = tf.placeholder('float', [batch, 2*n_rnn_dim], name='initial_state')

    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', seq_len_ph)
    tf.add_to_collection('inputs', initial_state_ph)

    net = tf.contrib.layers.fully_connected(
        inputs=obs_ph,
        num_outputs=16,
        activation_fn=tf.nn.elu,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        scope='fc1',
    )
    rnn_input = tf.expand_dims(net, 0)

    cell = tf.nn.rnn_cell.BasicLSTMCell(n_rnn_dim, state_is_tuple=False)
    # cell = tf.nn.rnn_cell.GRUCell(n_rnn_dim)

    seq_len = tf.shape(obs_ph)[:1]
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input,
                                                 sequence_length=seq_len,
                                                 initial_state=initial_state_ph,
                                                 time_major=False)

    rnn_outputs = tf.squeeze(rnn_outputs, 0)
    action_logits = tf.contrib.layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=n_actions,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='action_fc1',
    )
    action_probs = tf.nn.softmax(action_logits, name='action_probs')

    state_values = tf.contrib.layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=1,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='value_fc1',
    )
    state_values = tf.squeeze(state_values, -1)

    tf.add_to_collection('outputs', action_probs)
    tf.add_to_collection('outputs', state_values)

    # policy function
    def pi_func(obs_val, rnn_state_val):
        sess = tf.get_default_session()
        p, state_value_val, next_rnn_state_val = sess.run([action_probs, state_values, final_state], {
            obs_ph: [obs_val],
            initial_state_ph: [rnn_state_val],
        })
        return np.random.choice(n_actions, p=p[0]), state_value_val[0], next_rnn_state_val[0]

    # value function
    def v_func(obs_val, rnn_state_val):
        return state_values.eval(feed_dict={
            obs_ph: [obs_val],
            initial_state_ph: [rnn_state_val],
        })[0]

    return obs_ph, seq_len_ph, initial_state_ph, action_probs, state_values, final_state, \
    pi_func, v_func
