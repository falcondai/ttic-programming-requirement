from core import Env
from visualize import render_image
from Queue import deque
import numpy as np
import cv2
from gym import spaces

# common utilities

# image state transformers
class ScaleWrapper(Env):
    def __init__(self, env, scale, interpolation=cv2.INTER_LINEAR, render_fps=30.):
        assert isinstance(env.spec['observation_space'], spaces.Box)
        assert len(env.spec['observation_space'].shape) == 3

        ob_space = env.spec['observation_space']
        self.is_grayscale = ob_space.shape[-1] == 1
        h, w, c = ob_space.shape
        # keeping the aspect ratio
        sw, sh = int(w * scale), int(h * scale)
        self.size = (sw, sh)
        self.interpolation = interpolation
        self.env = env

        self.spec = dict(env.spec)
        low = np.min(ob_space.low)
        high = np.max(ob_space.high)
        self.spec['observation_space'] = spaces.Box(low=low, high=high, shape=(sh, sw, c))
        self.spec['id'] = '%s [%.2f scaled]' % (env.spec['id'], scale)
        self.scale = scale
        self.render_fps = render_fps
        self.obs = None

    def process_ob(self, ob):
        sob = cv2.resize(ob, self.size, interpolation=self.interpolation)
        if self.is_grayscale:
            # XXX cv2 automatically squeezes output?!
            return np.expand_dims(sob, -1)
        return sob

    def reset(self):
        obs = self.process_ob(self.env.reset())
        self.obs = obs
        return obs

    def step(self, action):
        ob, reward, done = self.env.step(action)
        obs = self.process_ob(ob)
        self.obs = obs
        return obs, reward, done

    def render(self):
        render_image(self.spec['id'], self.obs, self.render_fps, self.is_grayscale)

class GrayscaleWrapper(Env):
    def __init__(self, env, render_fps=30.):
        assert isinstance(env.spec['observation_space'], spaces.Box)
        assert len(env.spec['observation_space'].shape) == 3

        self.env = env
        self.process_ob = lambda ob: np.mean(ob, -1, keepdims=True)
        self.spec = dict(env.spec)
        ob_space = env.spec['observation_space']
        h, w, c = ob_space.shape
        low = np.min(ob_space.low)
        high = np.max(ob_space.high)
        self.spec['observation_space'] = spaces.Box(low=low, high=high, shape=(h, w, 1))
        self.spec['id'] = '%s [grayscale]' % env.spec['id']
        self.render_fps = render_fps
        self.obs = None

    def reset(self):
        obs = self.process_ob(self.env.reset())
        self.obs = obs
        return obs

    def step(self, action):
        ob, reward, done = self.env.step(action)
        obs = self.process_ob(ob)
        self.obs = obs
        return obs, reward, done

    def render(self):
        render_image(self.spec['id'], self.obs, self.render_fps, True)

class StackFrameWrapper(Env):
    def __init__(self, env, stack_frames=4, render_fps=30.):
        assert isinstance(env.spec['observation_space'], spaces.Box)
        assert len(env.spec['observation_space'].shape) == 3

        self.is_grayscale = env.spec['observation_space'].shape[-1] == 1
        self.env = env
        self.spec = dict(env.spec)
        self.spec['id'] = '%s [%i stacked frames]' % (env.spec['id'], stack_frames)
        ob_space = env.spec['observation_space']
        h, w, c = ob_space.shape
        low = np.min(ob_space.low)
        high = np.max(ob_space.high)
        self.spec['observation_space'] = spaces.Box(low=low, high=high, shape=(stack_frames, h, w, c))
        self.stack_frames = stack_frames
        self.render_fps = render_fps
        self.observation_queue = deque(maxlen=self.stack_frames)

    def reset(self):
        obs = self.env.reset()
        self.observation_queue = deque(np.zeros(self.spec['observation_space'].shape)[:-1], maxlen=self.stack_frames)
        self.observation_queue.append(obs)

        return np.asarray(self.observation_queue)

    def step(self, action):
        obs, reward, done = self.env.step(action)
        self.observation_queue.popleft()
        self.observation_queue.append(obs)

        return np.asarray(self.observation_queue), reward, done

    def render(self):
        fs = np.concatenate(self.observation_queue, 1)
        render_image(self.spec['id'], fs, self.render_fps, self.is_grayscale)

class MotionBlurWrapper(Env):
    def __init__(self, env, mix_coeff=0.6, render_fps=30.):
        assert isinstance(env.spec['observation_space'], spaces.Box)
        assert len(env.spec['observation_space'].shape) == 3

        self.is_grayscale = env.spec['observation_space'].shape[-1] == 1
        self.env = env
        self.spec = dict(env.spec)
        self.spec['id'] = '%s [motion blur]' % env.spec['id']
        self.mix_coeff = mix_coeff
        self.render_fps = render_fps
        self.last_observation = None

    def reset(self):
        self.last_observation = self.env.reset()
        return self.last_observation

    def step(self, action):
        obs, reward, done = self.env.step(action)
        # motion blur by taking exponential moving average
        self.last_observation = self.mix_coeff * obs + (1. - self.mix_coeff) * self.last_observation

        return self.last_observation, reward, done

    def render(self):
        render_image(self.spec['id'], self.last_observation, self.render_fps, self.is_grayscale)

# action transformers

class KeyMapWrapper(Env):
    def __init__(self, env, key_map):
        # assume that key_map is a list/tuple
        # mapping key i -> key_map[i]
        self.env = env
        self.spec = dict(env.spec)
        self.spec['id'] = '%s [key mapped]' % env.spec['id']
        self.spec['action_space'] = spaces.Discrete(len(key_map))
        self.key_map = key_map
        self.reset = env.reset
        self.render = env.render

    def step(self, action):
        return self.env.step(self.key_map[action])

# environment combination


if __name__ == '__main__':
    from core import test_env, GymEnv
    # env = GymEnv('PongDeterministic-v3')
    # env = GymEnv('SpaceInvadersDeterministic-v3')
    env = GymEnv('BreakoutDeterministic-v3')

    # env = ScaleWrapper(GrayscaleWrapper(env), scale=0.5, render_fps=10.)
    # slower
    # env = GrayscaleWrapper(ScaleWrapper(env, scale=42./160.))

    test_env(MotionBlurWrapper(env, render_fps=10.))
    # test_env(env, True)
    # test_env(GrayscaleWrapper(ScaleWrapper(env, scale=42./160.)), False)
    # test_env(GrayscaleWrapper(env))
    # test_env(StackFrameWrapper(GrayscaleWrapper(env)))
    # test_env(KeyMapWrapper(env, [2, 3]))
