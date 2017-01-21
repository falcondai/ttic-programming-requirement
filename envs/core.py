import numpy as np

class Env:
    def __init__(self):
        self.spec = {
            'id': '',
            'observation_shape': [],
            'action_size': [],
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
        if self.spec['timestep_limit'] != None and self.timestep > self.spec['timestep_limit']:
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

def test_env(env):
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

if __name__ == '__main__':
    # env = GymEnv('CartPole-v0')
    env = GymEnv('Breakout-v0')
    test_env(env)
