import tensorflow as tf
import numpy as np


def wrap_rnn_a3c():
    # policy function
    action = tf.multinomial(action_logits, 1)
    def pi_v_h_func(obs_val, rnn_state_val):
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

    zero_state = np.zeros((2, 1, n_rnn_dim))

    return obs_ph, initial_state_ph, \
    action_logits, state_values, final_state, \
    pi_v_h_func, v_func, zero_state
