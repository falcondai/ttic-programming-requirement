from core import GymEnv
from atari import get_atari_env
from doom import get_doom_env
from wrappers import GrayscaleWrapper, ScaleWrapper, MotionBlurWrapper

def get_env(env_id):
    '''env_id is a string of the format `prefix.env_name`'''
    # parse
    parts = env_id.split('.')
    gym_env_prefix = 'gym'
    atari_env_prefix = 'atari'
    doom_env_prefix = 'doom'

    # envs from OpenAI gym with no wrappers
    if parts[0] == gym_env_prefix:
        return GymEnv(parts[1])

    # Atari games
    if parts[0] == atari_env_prefix:
        return get_atari_env('.'.join(parts[1:]))

    # Doom envs
    if parts[0] == doom_env_prefix:
        return get_doom_env('.'.join(parts[1:]))
