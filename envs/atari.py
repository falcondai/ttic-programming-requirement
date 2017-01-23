import numpy as np

from core import GymEnv
from wrappers import GrayscaleWrapper, ScaleWrapper


def half_size_atari(env_id):
    # commonly used resizing grayscale
    # use atari games via OpenAI gym
    env = GymEnv(env_id)
    return ScaleWrapper(GrayscaleWrapper(env), scale=0.5)


if __name__ == '__main__':
    from core import test_env
    env = half_size_atari('Pong-v0')
    # env = MotionBlurAtariEnv('Pong-v0', scale=1.)
    test_env(env)
