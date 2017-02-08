from scipy.misc import imresize
from functools import partial
from Queue import deque
import numpy as np
import tensorflow as tf
import glob, os, time, itertools
import scipy.signal

# reward processing
def discount(x, gamma):
    # magic formula for computing gamma-discounted rewards
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# rollout
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

        obs, actions, rewards, terminals, vhats, info = [], [], [], [], [], {}
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

            terminals.append(done)

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
        yield obs + [observation], actions, rewards, terminals, vhats, info

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
        'timestep_limit': gym_env.spec.timestep_limit if 'timestep_limit' in dir(gym_env.spec) else 10**6,
        'action_size': gym_env.action_space.n,
        'observation_shape': gym_env.observation_space.shape,
    }
    step = lambda action: gym_env.step(action)[:3]
    return spec, step, gym_env.reset, gym_env.render

def scale_image(scale, interpolation, im):
    return imresize(im, scale, interp=interpolation)

def grayscale_image(im):
    return im.mean(axis=2, keepdims=True)

def scale_env(env, scale, interpolation):
    spec, step, reset, render = env
    spec['observation_shape'] = scale_image(scale, interpolation,
                                            np.zeros(
                                                spec['observation_shape']
                                                )
                                            ).shape
    env_reset = lambda : scale_image(scale, interpolation, reset())
    def env_step(action):
        im, reward, done = step(action)
        return grayscale_image(scale_image(scale, interpolation, im)), reward, done
    return spec, env_step, env_reset, render

def atari_env(env, scale, skip_frames):
    spec, step, reset, render = env
    spec['observation_shape'] = grayscale_image(scale_image(scale,
                                                            'bilinear',
                                                            np.zeros(
                                                                (160, 160, 3)
                                                                ))).shape

    # spec['observation_shape'] = grayscale_image(scale_image(scale,
    #                                                         'bilinear',
    #                                                         np.zeros(
    #                                                             spec['observation_shape']
    #                                                             ))).shape
    env_reset = lambda : grayscale_image(scale_image(scale, 'bilinear', reset()[34:34+160, :160]))
    def env_step(action):
        acc_reward = 0.
        for i in xrange(skip_frames):
            im, reward, done = step(action)
            im = im[34:34+160, :160]
            acc_reward += reward
            if done:
                break
        return grayscale_image(scale_image(scale, 'bilinear', im)), acc_reward, done
    return spec, env_step, env_reset, render

def use_render_state(gym_env, scale, interpolation='bilinear'):
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
