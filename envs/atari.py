from util import grayscale_image, scale_image
from core import Env
import gym

import numpy as np
from Queue import deque
import time

class StackAtariEnv(Env):
    def __init__(self, env_id, scale=42./160, interpolation='bilinear', stack_frames=4, render_fps=30.):
        # use atari games via OpenAI gym
        # TODO try using ALE directly?
        # via https://github.com/bbitmaster/ale_python_interface
        self.gym_env = gym.make(env_id)
        self.process_observation = lambda obs: grayscale_image(scale_image(scale, interpolation, obs))
        shape = self.process_observation(np.zeros(self.gym_env.observation_space.shape)).shape
        shape = [stack_frames] + list(shape)
        self.spec = {
            'observation_shape': shape,
            'action_size': self.gym_env.action_space.n,
            'timestep_limit': self.gym_env.spec.timestep_limit if 'timestep_limit' in dir(self.gym_env.spec) else None,
        }
        self.stack_frames = stack_frames
        self.scale = scale
        self.env_id = env_id
        self.render_fps = render_fps
        self.observation_queue = deque(maxlen=self.stack_frames)

    def reset(self):
        obs = self.process_observation(self.gym_env.reset())
        self.observation_queue = deque(np.zeros(self.spec['observation_shape'])[:-1], maxlen=self.stack_frames)
        self.observation_queue.append(obs)

        return self.observation_queue

    def step(self, action):
        ob, reward, done, info = self.gym_env.step(action)
        obs = self.process_observation(ob)
        self.observation_queue.popleft()
        self.observation_queue.append(obs)

        return self.observation_queue, reward, done

    def render(self):
        import cv2
        fs = np.hstack([f[:, :, 0] for f in self.observation_queue])
        cv2.imshow(self.env_id, np.asarray(fs, dtype='uint8'))
        # cv2.imshow(self.env_id, np.asarray(self.observation_queue[-1][:, :, 0], dtype='uint8'))
        cv2.waitKey(int(1000. / self.render_fps))

class MotionBlurAtariEnv(Env):
    def __init__(self, env_id, scale=42./160, interpolation='bilinear', mix_coeff=0.6, render_fps=30.):
        # use atari games via OpenAI gym
        self.gym_env = gym.make(env_id)
        self.process_observation = lambda obs: grayscale_image(scale_image(scale, interpolation, obs))
        shape = self.process_observation(np.zeros(self.gym_env.observation_space.shape)).shape
        self.spec = {
            'observation_shape': shape,
            'action_size': self.gym_env.action_space.n,
            'timestep_limit': self.gym_env.spec.timestep_limit if 'timestep_limit' in dir(self.gym_env.spec) else None,
        }
        self.mix_coeff = mix_coeff
        self.scale = scale
        self.env_id = env_id
        self.render_fps = render_fps
        self.last_observation = None

    def reset(self):
        obs = self.process_observation(self.gym_env.reset())
        self.last_observation = obs

        return self.last_observation

    def step(self, action):
        ob, reward, done, info = self.gym_env.step(action)
        obs = self.process_observation(ob)
        # motion blur by taking exponential moving average
        self.last_observation = self.mix_coeff * obs + (1. - self.mix_coeff) * self.last_observation

        return self.last_observation, reward, done

    def render(self):
        import cv2
        cv2.imshow(self.env_id, np.asarray(self.last_observation[:, :, 0], dtype='uint8'))
        cv2.waitKey(int(1000. / self.render_fps))


if __name__ == '__main__':
    from core import test_env
    env = StackAtariEnv('Pong-v0', scale=1., render_fps=100.)
    # env = MotionBlurAtariEnv('Pong-v0', scale=1.)
    print env.spec
    test_env(env)
