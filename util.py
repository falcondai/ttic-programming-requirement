from scipy.misc import imresize
import numpy as np
import tensorflow as tf
import glob, os, time, itertools, json
import scipy.signal

# reward processing
def discount(rewards, gamma):
    # magic formula for computing gamma-discounted rewards
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]

# returns processing
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

def vector_slice(A, B):
    ''' Returns values of rows i of A at column B[i]

    where A is a 2D Tensor with shape [None, D]
    and B is a 1D Tensor with shape [None]
    with type int32 elements in [0,D)

    Example:
      A =[[1,2], B = [0,1], vector_slice(A,B) -> [1,4]
          [3,4]]
    '''
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
