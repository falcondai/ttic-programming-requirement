from core import Env
from visualize import render_image

import numpy as np
import time, itertools, os
from gym import spaces

import vizdoom
from vizdoom import *

class DoomEnv(Env):
    def __init__(self, config_path, package_dir_as_root=True, repeat_action=4, screen_shape=(240, 320), use_grayscale=True, draw_hud=False, draw_weapon=True, use_depth=False, use_labels=False, use_automap=False, render_fps=60.):
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

        # game settings
        res_code = vars(ScreenResolution)['RES_%iX%i' % screen_shape[::-1]]
        self.game.set_screen_resolution(res_code)
        self.use_grayscale = use_grayscale
        if use_grayscale:
            self.game.set_screen_format(ScreenFormat.GRAY8)
        else:
            # XXX BGR24 actually returns RGB with vizdoom?!
            self.game.set_screen_format(ScreenFormat.BGR24)

        self.use_depth = use_depth
        self.game.set_depth_buffer_enabled(self.use_depth)

        self.use_automap = use_automap
        if self.use_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(True)
            self.game.set_automap_render_textures(False)
        else:
            self.game.set_automap_buffer_enabled(False)

        self.use_labels = use_labels
        self.game.set_labels_buffer_enabled(self.use_labels)

        self.game.set_render_hud(draw_hud)
        self.game.set_render_weapon(draw_weapon)
        self.game.set_window_visible(False)
        self.game.init()

        self.render_fps = render_fps
        # repeat the same action for a few frames
        self.repeat_action = repeat_action
        self.last_ob = None

        w, h = self.game.get_screen_width(), self.game.get_screen_height()
        c = 1 if self.use_grayscale else 3
        self.spec = {
            'id': 'doom',
            'observation_space': spaces.Box(low=0., high=255., shape=(h, w, c)),
            'action_space': spaces.Discrete(len(self.action_map)),
            'timestep_limit': self.game.get_episode_timeout(),
            'extra_variables': self.game.get_available_game_variables(),
        }

    def reset(self):
        self.game.new_episode()
        self._update_state()
        return self.last_ob

    def step(self, action):
        reward = self.game.make_action(self.action_map[action], self.repeat_action)
        terminal = self.game.is_episode_finished()
        if terminal:
            return self.last_ob, reward, terminal

        self._update_state()
        return self.last_ob, reward, terminal

    def _update_state(self):
        # TODO add support for other buffers
        ob = self.game.get_state().screen_buffer
        if self.use_grayscale:
            self.last_ob = np.expand_dims(ob, -1)
        else:
            self.last_ob = ob

    def render(self):
        # TODO show other buffers
        # if self.game.get_state():
        #     render_image('depth', np.expand_dims(self.game.get_state().depth_buffer, -1), self.render_fps, True)
        #     render_image('automap', np.expand_dims(self.game.get_state().automap_buffer, -1), self.render_fps, True)
        #     render_image('label', np.expand_dims(self.game.get_state().labels_buffer, -1), self.render_fps, True)
        render_image(self.spec['id'], self.last_ob, self.render_fps, self.use_grayscale)

def get_doom_env(env_id):
    parts = env_id.split('.')
    discrete_envs = [
        # a stationary target in a simple room
        'simpler_basic',
        'basic',
        'rocket_basic',
        'deadly_corridor',
        'defend_the_center',
        'defend_the_line',
        'health_gathering',
        'my_way_home',
        'predict_position',
        'take_cover',
    ]
    if parts[-1] in discrete_envs:
        cfg_path = os.path.join('scenarios', '%s.cfg' % parts[-1])

    return DoomEnv(cfg_path, screen_shape=(120, 160), repeat_action=4)


if __name__ == '__main__':
    from core import test_env
    import sys

    env_id = sys.argv[1]
    test_env(get_doom_env(env_id))
