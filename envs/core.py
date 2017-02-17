import numpy as np
import time

class Env:
    def __init__(self):
        self.spec = {
            'id': '',
            'observation_shape': [],
            'action_size': 0,
            'timestep_limit': 1,
        }

    def reset(self):
        # return observation
        raise NotImplementedError

    def step(self, action):
        # return observation, reward, done
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

class GymEnv(Env):
    def __init__(self, env_id):
        import gym
        self.env_id = env_id
        self.gym_env = gym.make(env_id)
        self.spec = {
            'id': env_id,
            'observation_shape': self.gym_env.observation_space.shape,
            'timestep_limit': self.gym_env.spec.timestep_limit if 'timestep_limit' in dir(self.gym_env.spec) else None,
            'action_size': self.gym_env.action_space.n,
        }
        self.timestep = 0

    def reset(self):
        self.timestep = 0
        return self.gym_env.reset()

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(action)
        self.timestep += 1
        # enforce timestep_limit
        if self.spec['timestep_limit'] != None and self.timestep == self.spec['timestep_limit']:
            done = True
        return obs, reward, done

    def render(self):
        return self.gym_env.render()


# utility functions
def repeat_action_wrapper(step, n_repeat):
    def env_step(action):
        acc_reward = 0.
        for i in xrange(n_repeat):
            obs, reward, done = step(action)
            acc_reward += reward
            if done:
                break
        return obs, acc_reward, done
    return env_step

def test_env(env, render=True):
    t0 = time.time()
    n = 0
    print env.spec
    n_action = env.spec['action_size']
    obs = env.reset()
    done = False
    episode_reward = 0.
    episode_length = 0
    print 'obs shape', np.shape(obs)
    while True:
        if done:
            obs = env.reset()
            print 'reward', episode_reward, 'length', episode_length
            episode_reward = 0.
            episode_length = 0
            n += 1
            if render:
                env.render()
        action = np.random.randint(n_action)
        obs, reward, done = env.step(action)
        episode_length += 1
        episode_reward += reward
        n += 1
        if render:
            env.render()
        t = time.time()
        # calculate FPS every 5 seconds
        if t - t0 > 5:
            print '%.2f fps' % (n * 1. / (t - t0))
            t0 = t
            n = 0

if __name__ == '__main__':
    import sys

    env_id = sys.argv[1]
    render = True if len(sys.argv) > 2 else False
    env = GymEnv(env_id)
    test_env(env, render)
