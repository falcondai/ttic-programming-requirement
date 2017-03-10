from scipy.misc import imresize
import numpy as np
import tensorflow as tf
import glob, os, time, itertools, json
import scipy.signal

# reward processing
def discount(rewards, gamma):
    # magic formula for computing gamma-discounted rewards
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]

# returns
def n_step_return(rewards, values, gamma, bootstrap_value, n_step=1):
    ''' computes n-step TD return '''
    n_step = min(n_step, len(rewards))
    returns = np.concatenate((values[n_step:], [bootstrap_value] * (n_step + 1)))
    for dt in xrange(n_step):
        returns[:-(n_step-dt)] = rewards[n_step-dt-1:] + gamma * returns[:-(n_step-dt)]
    return returns[:-1]

def td_return(rewards, values, gamma, bootstrap_value):
    ''' computes TD return, i.e. n-step TD return with n = 1'''
    return rewards + gamma * np.concatenate((values[1:], [bootstrap_value]))

def mc_return(rewards, gamma, bootstrap_value):
    ''' computes infinity-step return, i.e. MC return, with bootstraping state value.
    equivalent to setting n to larger than len(rewards) in n-step return '''
    return discount(np.concatenate((rewards, [bootstrap_value])), gamma)[:-1]

def lambda_return(rewards, values, gamma, td_lambda, bootstrap_value):
    td_error = td_return(rewards, values, gamma, bootstrap_value) - values
    lambda_error = discount(td_error, gamma * td_lambda)
    return lambda_error + values

# rollout generator
def partial_rollout(env_reset, env_step, pi_v_h_func, zero_state, n_ticks=None, env_render=None):
    done = True
    tick = 0
    while True:
        rollout_start = time.time()
        # on-policy rollout
        if done:
            # reset episode stats
            episode_len = 0
            episode_reward = 0.

            # reset the env
            observation = env_reset()

            if env_render != None:
                env_render()

            # initial rnn state
            h = zero_state
            h0 = h

        obs, actions, rewards, vhats, info = [], [], [], [], {}
        # sample some ticks for training
        for t in xrange(n_ticks) if n_ticks != None else itertools.count():
            obs.append(observation)
            # sample action according to policy
            action, v, h = pi_v_h_func(observation, h)
            actions.append(action)

            observation, reward, done = env_step(action)
            if env_render != None:
                env_render()

            tick += 1

            rewards.append(reward)
            vhats.append(v)

            episode_reward += reward
            episode_len += 1

            if done:
                # stop rollout at the end of the episode
                break

        # yield partial rollout
        info['rollout_dt'] = time.time() - rollout_start
        info['tick'] = tick
        if done:
            # report episode stats
            info['episode_len'] = episode_len
            info['episode_reward'] = episode_reward

        info['initial_state'] = h0
        info['final_state'] = h
        h0 = h

        # note the `obs` sequence has one extra final element than others
        # the final observation can be used for bootstraping reward-to-go
        yield obs + [observation], np.asarray(actions), np.asarray(rewards), np.asarray(vhats), done, info

def vector_slice(A, B):
    """ Returns values of rows i of A at column B[i]

    where A is a 2D Tensor with shape [None, D]
    and B is a 1D Tensor with shape [None]
    with type int32 elements in [0,D)

    Example:
      A =[[1,2], B = [0,1], vector_slice(A,B) -> [1,4]
          [3,4]]
    """
    linear_index = (tf.shape(A)[1] * tf.range(0, tf.shape(A)[0]))
    linear_A = tf.reshape(A, [-1])
    return tf.gather(linear_A, B + linear_index)

def mask_slice(x, a, depth=None):
    ''' same output as `vector_slice` but implement via one-hot masking
    '''
    if depth == None:
        depth = x.get_shape()[1]
    mask = tf.one_hot(a, depth, axis=-1, dtype=tf.float32)
    return tf.reduce_sum(x * mask, 1)

def get_optimizer(opt_name, learning_rate, momentum):
    if opt_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif opt_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, centered=True)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    return optimizer

def restore_save_hyperparameters(checkpoint_path, args):
    # try to find and restore hyperparameters
    hp_path = os.path.join(checkpoint_path, 'hyperparameters.json')
    if os.path.exists(hp_path):
        print '* found saved hyperparatermeters at %s' % hp_path
        saved = json.load(open(hp_path))
        for k, v in saved.iteritems():
            if not hasattr(args, k):
                # restore the missing hyperparameters
                setattr(args, k, v)
            else:
                # overwrite the saved hyperparameters
                saved[k] = v
    else:
        print '* no saved hyperparatermeters found'

    json.dump(vars(args), open(hp_path, 'wb'))

    return args

# tensorflow utility
def test_restore_vars(sess, checkpoint_path, meta_path):
    ''' Restore graph from metagraph then values from checkpoint '''
    print '* using metagraph from %s' % meta_path
    saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
    print '* restoring from %s' % checkpoint_path
    saver.restore(sess, checkpoint_path)
    return True

def get_current_run_id(checkpoint_dir):
    paths = glob.glob('%s/hyperparameters.*.json' % checkpoint_dir)
    if len(paths) == 0:
        return 0
    return sorted(map(lambda p: int(p.split('.')[-2]), paths))[-1] + 1

def restore_vars(saver, sess, checkpoint_dir, restart=False):
    ''' Restore values OR initialize '''
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    if not restart:
        path = tf.train.latest_checkpoint(checkpoint_dir)
        if path is None:
            print '* no existing checkpoint found'
            return False
        else:
            print '* restoring from %s' % path
            saver.restore(sess, path)
            return True
    print '* overwriting checkpoints at %s' % checkpoint_dir
    return False
