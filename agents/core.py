import argparse

class Agent(object):
    def __init__(self, env_spec):
        self.spec = {
            'deterministic': True, # agent models a deterministic policy
        }
        raise NotImplementedError

    def act(self, ob):
        # possibly use history, returns an action
        raise NotImplementedError

class StatefulAgent(Agent):
    ''' an agent that keeps an internal state '''
    def __init__(self, env_spec):
        self.spec = {
            'deterministic': True,
        }
        self._history_state = self.zero_state = 0.
        raise NotImplementedError

    def initial_state(self):
        return self.zero_state

    def history_state(self):
        return self._history_state

    def reset(self, history=None):
        if history:
            self._history_state = history
        else:
            self._history_state = self.zero_state

class RandomAgent(Agent):
    def __init__(self, env_spec):
        self.env_spec = env_spec
        self.spec = {
            'deterministic': False,
            'use_history': False,
        }

    def act(self, ob=None):
        return self.env_spec['action_space'].sample()

class Trainer(object):
    @staticmethod
    def parser():
        return argparse.ArgumentParser()

    def __init__(self, env, build_model, args):
        ''' init with environment and arguments parsed by parser '''
        raise NotImplementedError

    def setup(self):
        ''' setup rollout provider, queue, etc '''
        raise NotImplementedError

    def train(self, sess):
        ''' consume a (partial) rollout and update model '''
        raise NotImplementedError


if __name__ == '__main__':
    from envs import test_env, get_env

    env = get_env('gym.BipedalWalker-v2')
    agent = RandomAgent(env.spec)
    ob = env.reset()
    while True:
        action = agent.act(ob)
        ob, reward, done = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            print 'new episode'
