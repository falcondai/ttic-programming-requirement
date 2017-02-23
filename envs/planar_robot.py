from core import Env
import numpy as np
import itertools
from gym import spaces

def distance(x, y):
    return np.linalg.norm(x - y)

class PlanarReach(Env):
    def __init__(self, dtheta=0.05, goal_threshold=0.4, timestep_limit=800, n_joints=2):
        self.n_joints = n_joints
        self.links = np.ones(self.n_joints) * 2.
        self.positions = np.zeros(self.n_joints)
        self.goal_threshold = goal_threshold
        self.dtheta = dtheta
        self.timestep_limit = timestep_limit
        # O(n) discrete actions
        self.action_map = np.concatenate((dtheta * np.eye(self.n_joints), -dtheta * np.eye(self.n_joints), np.zeros((1, self.n_joints))))
        # O(n^2) discrete actions
        # self.action_map = np.asarray(list(itertools.product([-1., 0., 1.], repeat=self.n_joints))) * self.dtheta

        self.spec = {
            'id': 'planar-robot-%i' % self.n_joints,
            'observation_space': spaces.Box(low=0, high=0, shape=(3 * self.n_joints + 2 * 2,)),
            'action_space': spaces.Discrete(len(self.action_map)),
            'timestep_limit': self.timestep_limit,
        }

        self.viewer = None

    def _goal_distance(self):
        return distance(self.goal, self._get_endpoint_position())

    def _get_endpoint_position(self):
        # forward kinematics
        acc_pos = np.cumsum(self.positions)
        return np.sum([np.cos(acc_pos) * self.links, np.sin(acc_pos) * self.links], axis=1)

    def _check_goal(self):
        return self._goal_distance() < self.goal_threshold

    def _get_ob(self):
        # return both sin, cos of all positions
        return np.concatenate([
            np.cos(self.positions),
            np.sin(self.positions),
            self.positions,
            self.goal,
            self.goal - self._get_endpoint_position(),
            ])

    def reset(self):
        self.tick = 0

        # generate a random goal
        self.positions = np.random.rand(self.n_joints) * 2 * np.pi
        if self.n_joints == 1:
            r = np.sum(self.links)
        else:
            r = np.random.rand() * np.sum(self.links)
        theta = np.random.rand() * 2. * np.pi
        x, y = np.cos(theta), np.sin(theta)
        self.goal = np.asarray([x, y]) * r

        return self._get_ob()

    def step(self, action):
        # TODO implement dynamics with Box2D
        self.tick += 1
        self.positions = np.mod(self.positions + self.action_map[action], 2. * np.pi)

        # check termination conditions
        reached_goal = self._check_goal()
        done = reached_goal or self.tick == self.spec['timestep_limit']

        return self._get_ob(), -self._goal_distance(), done

    def render(self):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(320, 320)
            half_width = np.sum(self.links) + 1.
            self.viewer.set_bounds(-half_width, half_width, -half_width, half_width)

            # draw goal
            circle = rendering.make_circle(self.goal_threshold)
            circle.set_color(0.8, 0.2, 0.2)
            self.goal_tf = rendering.Transform()
            circle.add_attr(self.goal_tf)
            self.viewer.add_geom(circle)

            # draw arm
            self.frames = [rendering.Transform(translation=(0.0, 0.0))] + [rendering.Transform(translation=(link, 0.0)) for link in self.links[:-1]]
            for i, link in enumerate(self.links):
                rod = rendering.make_capsule(link, 0.2)
                rod.set_color(0., .5, .2)
                for j in xrange(i+1):
                    rod.add_attr(self.frames[i-j])
                self.viewer.add_geom(rod)

        self.goal_tf.set_translation(*self.goal)
        for i, (pos, link) in enumerate(zip(self.positions, self.links)):
            self.frames[i].set_rotation(pos)

        return self.viewer.render()

def get_planar_robot_env(env_id):
    parts = env_id.split('.')
    n_joints = int(parts[0]) if len(parts) > 0 else 1
    return PlanarReach(n_joints=n_joints)

if __name__ == '__main__':
    from core import test_env
    import sys
    env = get_planar_robot_env(sys.argv[1])
    test_env(env)
