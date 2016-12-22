import numpy as np
from util import passthrough, grayscale_image, scale_image

def atari(env, scale=0.2, interpolation='bilinear', skip_frames=4):
    spec, step, reset, render = env

    # common processing on atari game envs:
    # 0. scale down the images
    # 1. convert to grayscale
    # 2. skip a certain number of frames
    process_observation = lambda obs: grayscale_image(scale_image(scale, interpolation, obs))

    spec['observation_shape'] = process_observation(np.zeros( spec['observation_shape'])).shape

    env_reset = lambda : process_observation(reset())
    def env_step(action):
        acc_reward = 0.
        for i in xrange(skip_frames):
            im, reward, done = step(action)
            acc_reward += reward
            if done:
                break
        return process_observation(im), acc_reward, done
    return spec, env_step, env_reset, render

if __name__ == '__main__':
    import gym
    env = gym.make('Pong-v0')
    spec, step, reset, render = atari(passthrough(env))
    n_action = spec['action_size']
    done = True
    while True:
        if done:
            reset()
            render()
        action = np.random.randint(n_action)
        print action
        _, _, done = step(action)
        render()
