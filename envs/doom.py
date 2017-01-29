from core import Env
from visualize import render_image
import cv2

import numpy as np
import time, itertools, os

import vizdoom
from vizdoom import *

class DoomEnv(Env):
    def __init__(self, config_path, package_dir_as_root=True, render_fps=30.):
        self.game = DoomGame()
        if package_dir_as_root:
            dirname = os.path.dirname(vizdoom.__file__)
            self.config_path = os.path.join(dirname, config_path)
        else:
            self.config_path = config_path
        print 'loading config from', self.config_path
        self.game.load_config(self.config_path)

        buttons = self.game.get_available_buttons()
        self.action_map = [list(press) for press in itertools.product([0, 1], repeat=len(buttons))]

        w, h = self.game.get_screen_width(), self.game.get_screen_height()

        # XXX BGR24 actually returns RGB with vizdoom?!
        self.game.set_screen_format(ScreenFormat.BGR24)
        self.game.set_window_visible(False)
        self.game.init()

        self.render_fps = render_fps
        self.last_ob = None

        self.spec = {
            'id': 'doom',
            'observation_shape': (h, w, 3),
            'action_size': len(self.action_map),
            'timestep_limit': self.game.get_episode_timeout(),
            'extra_variables': self.game.get_available_game_variables(),
        }

    def reset(self):
        self.game.new_episode()
        ob = self.game.get_state().screen_buffer
        self.last_ob = ob
        return ob

    def step(self, action):
        reward = self.game.make_action(self.action_map[action])
        terminal = self.game.is_episode_finished()
        if terminal:
            return self.last_ob, reward, terminal
        next_ob = self.game.get_state().screen_buffer
        self.last_ob = next_ob
        return next_ob, reward, terminal

    def render(self):
        render_image(self.spec['id'], self.last_ob, self.render_fps, False)

if __name__ == '__main__':
    from core import test_env, repeat_action_wrapper
    from wrappers import ScaleWrapper, GrayscaleWrapper
    env = DoomEnv('scenarios/basic.cfg')
    # test_env(env)
    env = ScaleWrapper(env, 0.5)
    test_env(env, True)
