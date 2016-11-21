from scipy.misc import imresize
from functools import partial

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
