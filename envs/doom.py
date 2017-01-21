from util import passthrough, grayscale_image, scale_image
from core import Env
import gym
import cv2

import numpy as np
from Queue import deque
import time


if __name__ == '__main__':
    # env = StackAtariEnv('Pong-v0')
    env = MotionBlurAtariEnv('Pong-v0', scale=1.)
    print env.spec
    n_action = env.spec['action_size']
    done = True
    while True:
        if done:
            obs = env.reset()
            env.render()
        action = np.random.randint(n_action)
        obs, reward, done = env.step(action)
        print np.shape(obs)
        print action, reward
        env.render()
