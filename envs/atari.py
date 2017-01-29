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
    import sys

    # env = half_size_atari('Pong-v0')
    # env = MotionBlurAtariEnv('Pong-v0', scale=1.)
    env_id = sys.argv[1]
    # env = half_size_atari(env_id)
    env = ScaleWrapper(GymEnv(env_id), 0.5)
    test_env(env)
