import tensorflow as tf
import numpy as np
import itertools

def build_model(observation_shape, n_actions, batch=None, n_fc_dim=32):
    assert len(observation_shape) == 1

    obs_ph = tf.placeholder('float', [batch] + list(observation_shape), name='observation')

    tf.add_to_collection('inputs', obs_ph)

    # TODO observation transformer (tiling, hashing, rbf, etc)
    # transform the input space into tiles
    unit_obs = (obs_ph - [[-1.2, -0.07]]) / [[1.8, 0.14]]

    k = 8
    ind = tf.floor(unit_obs[:, 0] * k) * k + tf.floor(unit_obs[:, 1] * k)
    net = tf.one_hot(tf.cast(ind, tf.int32), k**2, dtype=tf.float32)

    # net = tf.contrib.layers.fully_connected(
    #     # inputs=tf.Print(net, ['rbf', tf.reduce_max(net), tf.reduce_min(net)]),
    #     inputs=net,
    #     num_outputs=n_fc_dim,
    #     biases_initializer=tf.zeros_initializer(),
    #     weights_initializer=tf.contrib.layers.xavier_initializer(),
    #     activation_fn=tf.nn.elu,
    #     scope='fc1',
    # )

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

    # policy function
    action = tf.multinomial(action_logits, 1)
    def pi_v_func(obs_val, history):
        sess = tf.get_default_session()
        action_val, state_value_val = sess.run([action, state_values], {
            obs_ph: [obs_val],
        })
        return action_val[0, 0], state_value_val[0], history

    # value function
    def v_func(obs_val, history):
        v = state_values.eval(feed_dict={
            obs_ph: [obs_val],
        })[0]
        print 'obs', obs_val, 'v', v
        return v

    zero_state = None

    return obs_ph, None, \
    action_logits, state_values, None, \
    pi_v_func, v_func, zero_state
