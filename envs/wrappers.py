from core import Env
from Queue import deque
import numpy as np
import cv2


# image state transformers

class ScaleWrapper(Env):
    def __init__(self, env, scale, interpolation=cv2.INTER_LINEAR, render_fps=30.):
        assert len(env.spec['observation_shape']) == 3
        h, w = env.spec['observation_shape'][:2]
        # keeping the aspect ratio
        sw, sh = int(w * scale), int(h * scale)
        size = (sw, sh)
        self.env = env
        self.process_ob = lambda ob: cv2.resize(ob, size, interpolation=interpolation)

        self.spec = dict(env.spec)
        self.spec['observation_shape'] = (sh, sw, env.spec['observation_shape'][-1])
        self.spec['id'] = '%s [%.2f scaled]' % (env.spec['id'], scale)
        self.scale = scale
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
        cv2.imshow(self.spec['id'], np.asarray(self.obs, dtype='uint8'))
        cv2.waitKey(int(1000. / self.render_fps))

class GrayscaleWrapper(Env):
    def __init__(self, env, render_fps=30.):
        assert len(env.spec['observation_shape']) == 3
        self.env = env
        self.process_ob = lambda ob: np.mean(ob, -1, keepdims=True)
        shape = self.process_ob(np.zeros(env.spec['observation_shape'])).shape
        self.spec = dict(env.spec)
        self.spec['observation_shape'] = shape
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
        cv2.imshow(self.spec['id'], np.asarray(self.obs, dtype='uint8'))
        cv2.waitKey(int(1000. / self.render_fps))

class StackFrameWrapper(Env):
    def __init__(self, env, stack_frames=4, render_fps=30.):
        assert len(env.spec['observation_shape']) == 3
        self.env = env
        shape = [stack_frames] + list(env.spec['observation_shape'])
        self.spec = dict(env.spec)
        self.spec['id'] = '%s [%i stacked frames]' % (env.spec['id'], stack_frames)
        self.spec['observation_shape'] = shape
        self.stack_frames = stack_frames
        self.render_fps = render_fps
        self.observation_queue = deque(maxlen=self.stack_frames)

    def reset(self):
        obs = self.env.reset()
        self.observation_queue = deque(np.zeros(self.spec['observation_shape'])[:-1], maxlen=self.stack_frames)
        self.observation_queue.append(obs)

        return np.asarray(self.observation_queue)

    def step(self, action):
        obs, reward, done = self.env.step(action)
        self.observation_queue.popleft()
        self.observation_queue.append(obs)

        return np.asarray(self.observation_queue), reward, done

    def render(self):
        fs = np.concatenate(self.observation_queue, 1)
        cv2.imshow(self.spec['id'], np.asarray(fs, dtype='uint8'))
        cv2.waitKey(int(1000. / self.render_fps))

class MotionBlurWrapper(Env):
    def __init__(self, env, mix_coeff=0.6, render_fps=30.):
        assert len(env.spec['observation_shape']) == 3
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
        cv2.imshow(self.spec['id'], np.asarray(self.last_observation, dtype='uint8'))
        cv2.waitKey(int(1000. / self.render_fps))

# action transformers

class KeyMapWrapper(Env):
    def __init__(self, env, key_map):
        # assume that key_map is a list/tuple
        # mapping key i -> key_map[i]
        self.env = env
        self.spec = dict(env.spec)
        self.spec['id'] = '%s [key mapped]' % env.spec['id']
        self.spec['action_size'] = len(key_map)
        self.key_map = key_map
        self.reset = env.reset
        self.render = env.render

    def step(self, action):
        return self.env.step(self.key_map[action])

# environment combination


if __name__ == '__main__':
    from core import test_env, GymEnv
    env = GymEnv('PongDeterministic-v3')
    test_env(ScaleWrapper(GrayscaleWrapper(env), scale=42./160.), False)
    # test_env(GrayscaleWrapper(ScaleWrapper(env, scale=42./160.)), False)
    # test_env(GrayscaleWrapper(env))
    # test_env(StackFrameWrapper(GrayscaleWrapper(env)))
    # test_env(MotionBlurWrapper(env))
    # test_env(KeyMapWrapper(env, [2, 3]))
