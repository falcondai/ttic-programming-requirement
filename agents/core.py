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
        raise NotImplementedError

    def act(self, ob):
        # use self.history_state() to return an action
        raise NotImplementedError

    def initial_state(self):
        # this should return the zero state
        raise NotImplementedError

    def history_state(self):
        # this should return the current state
        raise NotImplementedError

    def reset(self, history=None):
        # this should reset the internal history state to `history` or self.initial_state() by default
        raise NotImplementedError

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
