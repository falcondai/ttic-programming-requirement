import tensorflow as tf
import numpy as np

def build_model(observation_shape, n_actions, batch=None, n_cnn_layers=4, n_rnn_dim=256):
    obs_ph = tf.placeholder('float', [batch] + list(observation_shape),
                            name='observation')
    initial_state_ph = tf.placeholder('float', [batch, n_rnn_dim], name='initial_state')
    tf.add_to_collection('inputs', obs_ph)
    tf.add_to_collection('inputs', initial_state_ph)

    net = obs_ph / 255.
    for i in xrange(n_cnn_layers):
        net = tf.contrib.layers.convolution2d(
            inputs=net,
            num_outputs=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv%i' % (i+1),
        )
    net = tf.contrib.layers.flatten(net)

    # rnn
    rnn_input = tf.expand_dims(net, 0)
    cell = tf.contrib.rnn.GRUBlockCell(n_rnn_dim)

    seq_len = tf.shape(obs_ph)[:1]
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input, seq_len,
                                                 initial_state_ph)
    rnn_outputs = tf.squeeze(rnn_outputs, 0)

    # prediction outputs
    action_logits = tf.contrib.layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=n_actions,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='action_fc1',
    )

    state_values = tf.contrib.layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=1,
        biases_initializer=tf.zeros_initializer,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=None,
        scope='value_fc1',
    )
    state_values = tf.squeeze(state_values, -1)

    tf.add_to_collection('outputs', action_logits)
    tf.add_to_collection('outputs', state_values)
    tf.add_to_collection('outputs', final_state)

    # policy function
    action = tf.multinomial(action_logits, 1)
    def pi_v_func(obs_val, rnn_state_val):
        sess = tf.get_default_session()
        action_val, state_value_val, next_rnn_state_val = sess.run([action, state_values, final_state], {
            obs_ph: [obs_val],
            initial_state_ph: rnn_state_val,
        })
        return action_val[0, 0], state_value_val[0], next_rnn_state_val

    # value function
    def v_func(obs_val, rnn_state_val):
        return state_values.eval(feed_dict={
            obs_ph: [obs_val],
            initial_state_ph: rnn_state_val,
        })[0]

    zero_state = np.zeros((1, n_rnn_dim))

    return obs_ph, initial_state_ph, \
    action_logits, state_values, final_state, \
    pi_v_func, v_func, zero_state
