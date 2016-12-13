from scipy.misc import imresize
from functools import partial
from Queue import deque
import numpy as np
import tensorflow as tf
import glob, os

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

# define an environment as a 4-tuple
# {
#     spec: {
#         observation_shape,
#         timestep_limit,
#         action_size,
#     },
#     step: action |-> state, reward, done,
#     reset: |-> state,
#     render: |-> ,
# }

def passthrough(gym_env):
    '''use gym environment as is'''
    spec = {
        'timestep_limit': gym_env.spec.timestep_limit,
        'action_size': gym_env.action_space.n,
        'observation_shape': gym_env.observation_space.shape,
    }
    step = lambda action: gym_env.step(action)[:3]
    return spec, step, gym_env.reset, gym_env.render

def scale_image(scale, interpolation, im):
    return imresize(im, scale, interp=interpolation)

def use_render_state(gym_env, scale, interpolation='nearest'):
    '''use gym environment's rendered image as observation variable'''
    si = partial(scale_image, scale, interpolation)
    observation_shape = si(gym_env.render('rgb_array')).shape
    spec = {
        'timestep_limit': gym_env.spec.timestep_limit,
        'action_size': gym_env.action_space.n,
        'observation_shape': observation_shape,
    }

    def reset():
        gym_env.reset()
        return si(gym_env.render('rgb_array'))

    def step(action):
        obs, reward, done, info = gym_env.step(action)
        return si(gym_env.render('rgb_array')), reward, done

    return spec, step, reset, gym_env.render

def pad_zeros(obs, n_obs_ticks):
    return [np.zeros(obs[0].shape)] * (n_obs_ticks - 1) + obs

def duplicate_obs(observations, n_obs_ticks):
    obs_q = []
    l = len(observations)
    for i in xrange(n_obs_ticks):
        obs_q.append(observations[i:l-n_obs_ticks+i+1])
    return np.concatenate(obs_q, axis=-1)

def rollout(behavior_policy, env_spec, env_step, env_reset,
            env_render=None, n_obs_ticks=1):
    '''rollout based on behavior policy from an environment'''
    # pad the first observation with zeros
    obs = env_reset()
    obs_q = deque(pad_zeros([obs], n_obs_ticks), n_obs_ticks)

    observations, actions, rewards = [], [], []
    done = False
    t = 0
    while not done and t < env_spec['timestep_limit']:
        policy_input = np.concatenate(obs_q, axis=-1)
        action_probs = behavior_policy(policy_input)
        action = np.random.choice(env_spec['action_size'], p=action_probs)
        obs_q.popleft()
        observations.append(obs)
        actions.append(action)
        obs, reward, done = env_step(action)
        rewards.append(reward)
        obs_q.append(obs)
        if env_render != None:
            env_render()
        t += 1
    return observations, actions, rewards

# policy modifiers
def to_greedy(policy_prob_func, obs):
    ps = policy_prob_func(obs)
    z = np.zeros(ps.shape)
    z[np.argmax(ps)] = 1.
    return z

def to_epsilon_greedy(epsilon, policy_prob_func, obs):
    ps = policy_prob_func(obs)
    z = np.zeros(ps.shape) + epsilon / len(ps)
    z[np.argmax(ps)] += 1. - epsilon
    return z

# tensorflow utility
def test_restore_vars(sess, checkpoint_path, meta_path):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, checkpoint_path)

    print '* restoring from %s' % checkpoint_path
    print '* using metagraph from %s' % meta_path
    saver.restore(sess, checkpoint_path)
    return True

def get_current_run_id(checkpoint_dir):
    paths = glob.glob('%s/hyperparameters.*.json' % checkpoint_dir)
    if len(paths) == 0:
        return 0
    return sorted(map(lambda p: int(p.split('.')[-2]), paths))[-1] + 1

def restore_vars(saver, sess, checkpoint_dir, restart=False):
    ''' Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. '''
    sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])

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
