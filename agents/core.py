import argparse

class Agent(object):
    def __init__(self, env_spec):
        self.spec = {
            'deterministic': True, # agent models a deterministic policy
            'use_history': True, # policy uses history as input
        }
        raise NotImplementedError

    def act(self, ob, history=None):
        # possibly use history
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, env_spec):
        self.env_spec = env_spec
        self.spec = {
            'deterministic': False,
            'use_history': False,
        }

    def act(self, ob=None, history=None):
        return self.env_spec['action_space'].sample()

class Trainer(object):
    @staticmethod
    def parser():
        return argparse.ArgumentParser()

    def __init__(self, env, build_model, args):
        ''' init with environment and arguments parsed by parser '''
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
