from core import Env
from motion_routine import Trajectory

import itertools, time

import rospy
import baxter_interface
import actionlib
from baxter_pykdl import baxter_kinematics

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from baxter_core_msgs.msg import EndpointState
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from quaternion import Quaternion as Quat

import cv2
from cv_bridge import CvBridge
import numpy as np

def get_ik_joints_linear(initial_position, target_position, n_steps, limb):
    ns = "ExternalTools/%s/PositionKinematicsNode/IKService" % limb
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest(seed_mode=SolvePositionIKRequest.SEED_CURRENT)
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')

    x0, y0, z0 = initial_position
    x1, y1, z1 = target_position

    # print initial_orientation
    # q0 = initial_orientation
    # print 'q0', q0.as_v_theta()
    # q1 = target_orientation
    # print 'q1', q1, q1.as_v_theta()

    # linear interpolate between current pose and target pose
    for i in xrange(n_steps):
        t = (i + 1) * 1. / n_steps
        x = (1. - t) * x0 + t * x1
        y = (1. - t) * y0 + t * y1
        z = (1. - t) * z0 + t * z1

        # q = (q1 * q0 ** -1.) ** t * q0
        qw, qx, qy, qz = Quat.from_v_theta([0,0,1], 0.4).as_wxyz()
        # print q.as_v_theta()

        pose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point( x=x, y=y, z=z, ),
                # endeffector pointing down
                orientation=Quaternion(x=qw, y=qz, z=qy, w=qx),
            ),
        )
        ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return []
    #
    # js = []
    # # control w2 separately from other joints
    # for i, (v, j) in enumerate(zip(resp.isValid, resp.joints)):
    #     t = (i + 1) * 1. / n_steps
    #     if v:
    #         w2 = (1. - t) * initial_w2 + t * target_w2
    #         j.position = j.position[:-1] + (w2,)
    #         js.append(j)
    return zip(resp.isValid, resp.joints)

def get_linear_trajectory_from_current(arm, target_x, target_y, target_z, n_steps, duration):
    limb = arm.name
    traj = Trajectory(limb)
    current_pose = arm.endpoint_pose()
    current_position = current_pose['position']
    dt_step = duration * 1. / n_steps
    js = get_ik_joints_linear((current_position.x, current_position.y, current_position.z), (target_x, target_y, target_z), n_steps, limb)
    # if len(js) < n_steps:
    #     return None
    # print js[0]
    if not js[0]:
        print 'no trajectory found'
    for j, (v, joint) in enumerate(js):
        if v:
            traj.add_point(joint.position, dt_step * (j + 1))
            # print joint.position, dt_step * (j + 1)
    return traj

def execute_linear(arm, target_x, target_y, target_z, n_steps=1, duration=4., timeout=10.):
    traj = get_linear_trajectory_from_current(arm, target_x, target_y, target_z, n_steps, duration)
    if traj == None:
        return None
    traj.start()
    traj.wait(timeout)
    return traj


class BaxterReachEnv(Env):
    def __init__(self, reach_threshold=0.2, reward_per_sec=-1., move_dx=0.05, goal_origin=[1.04, 0.17, 0.30]):
        # setup
        rospy.init_node('reach_task_env')
        rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        rs.enable()

        # baxter_interface.CameraController('left_hand_camera').close()
        # baxter_interface.CameraController('right_hand_camera').close()
        head_camera = baxter_interface.CameraController('head_camera')
        head_camera.open()
        head_camera.resolution = (1280, 800)

        self.pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1)
        self.left_arm = baxter_interface.Limb('left')
        self.right_arm = baxter_interface.Limb('right')
        bridge = CvBridge()
        self.reach_threshold = reach_threshold
        print 'setup completed'

        # reset states
        self.head_camera_img = None
        self.right_hand_pos = np.asarray(self.right_arm.endpoint_pose()['position'])
        self.transient_reward = 0.
        self.last_updated_at = rospy.get_time()
        self.terminal = False

        # move left hand to goal
        self.right_arm_kin = baxter_kinematics('right')
        self.action_dirs = np.asarray([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ])
        self.move_dx = move_dx
        self.reward_per_sec = reward_per_sec
        self.joint_names = self.right_arm.joint_names()
        # print self.left_arm.endpoint_pose()
        # print right_arm_kin.inverse_kinematics(self.goal_pos)
        # left_arm.move_to()

        # execute_linear(left_arm, x0, y0, initial_z, Quat.from_v_theta([0,0,1], np.random.rand() * 2 * np.pi), n_steps=1, duration=5.)

        self.reset()

        # subscribe to robot states
        def update_image(msg):
            self.head_camera_img = img = bridge.imgmsg_to_cv2(msg, 'bgr8')
            # print self.head_camera_img.shape
            # cv2.imwrite('test.png', self.head_camera_img)
            msg = bridge.cv2_to_imgmsg(cv2.resize(img, (640, 400)), encoding='rgb8')
            self.pub.publish(msg)

        rospy.Subscriber('/cameras/head_camera/image', Image, update_image)

        def update_hand(msg):
            # p = msg.pose.position
            # self.right_hand_pos = np.asarray([p.x, p.y, p.z])
            p = self.right_arm.endpoint_pose()['position']
            self.right_hand_pos = np.asarray([p.x, p.y, p.z])
            dt = rospy.get_time() - self.last_updated_at
            self.transient_reward += self.reward_per_sec * dt
            self.last_updated_at = rospy.get_time()
            # print self.right_hand_pos, np.linalg.norm(self.goal_pos - self.right_hand_pos), self.terminal, self.transient_reward
            if self.check_reach():
                self.terminal = True
            # self.render()
        rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState, update_hand)

        self.spec = {
            'id': 'baxter-reach',
            'observation_shape': (800, 1280, 3),
            'action_size': 6,
        }

    def distance_to_goal(self):
        return np.linalg.norm(self.right_hand_pos - self.left_arm.endpoint_pose()['position'])

    def check_reach(self):
        if self.distance_to_goal() < self.reach_threshold:
            return True
        return False

    # def setup(self):
    #     self.goal_pos = self.left_arm.endpoint_pose()['position']
    #     self.right_arm.move_to_neutral()

    def reset(self):
        # self.right_arm.move_to_neutral()
        # self.goal_pos = np.asarray(self.left_arm.endpoint_pose()['position'])
        # js = self.right_arm_kin.inverse_kinematics(pos)
        # x = dict(zip(self.joint_names, js))
        # print x
        # self.right_arm.move_to_joint_positions(x)
        return self.head_camera_img

    def step(self, action):
        new_pos = self.right_hand_pos + self.action_dirs[action] * self.move_dx
        print 'new', self.right_hand_pos, new_pos
        execute_linear(self.right_arm, *new_pos, duration=1.)
        reward = self.transient_reward
        self.transient_reward = 0.
        return self.head_camera_img, reward, self.terminal

    def render(self):
        print self.right_hand_pos, self.distance_to_goal(), self.check_reach()

def distance(x, y):
    return np.linalg.norm(x - y)

class BaxterIkEnv(Env):
    def __init__(self, limb='left', dtheta=0.05, step_dt=1./30., goal_threshold=0.1, timestep_limit=300):
        assert limb in ['left', 'right']

        # setup
        rospy.init_node('ik_env_%s' % limb)
        rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        rs.enable()

        self.limb = limb
        self.inner_led = baxter_interface.DigitalIO('%s_inner_light' % limb)
        self.outer_led = baxter_interface.DigitalIO('%s_outer_light' % limb)
        self.arm = baxter_interface.Limb(limb)
        self.joint_names = self.arm.joint_names()
        n_joints = len(self.joint_names)
        self.dtheta = dtheta
        # large, simultaneous movement at each joint
        # self.action_map = [np.asarray(d) for d in itertools.product([-self.dtheta, +self.dtheta], repeat=7)]
        self.action_map = list(np.eye(n_joints) * dtheta) + list(np.eye(n_joints) * -dtheta)
        self.action_map = [np.zeros(7)] + self.action_map
        self.goal_threshold = goal_threshold
        self.step_dt = step_dt

        self.spec = {
            'id': 'baxter-ik-%s' % self.limb,
            'observation_shape': (n_joints + 3,),
            'action_size': len(self.action_map),
            'timestep_limit': timestep_limit,
        }

    def _get_state(self):
        x = self.arm.joint_angles()
        return np.asarray([x[k] for k in self.joint_names])

    def _get_endpoint_position(self):
        return np.asarray(list(self.arm.endpoint_pose()['position']))

    def _check_goal(self):
        # j = dict(zip(self.joint_names, self._get_state()))
        # p = self.kin.forward_position_kinematics(j)[:3]
        return distance(self.goal, self._get_endpoint_position()) < self.goal_threshold

    def reset(self, goal_distance=0.2):
        assert self.goal_threshold < goal_distance
        # randomly sample a goal near the initial hand position
        # `goal_distance` limits how far the goal can be from the hand
        r = np.random.rand() * (goal_distance - self.goal_threshold) + self.goal_threshold
        a, b = np.pi * np.random.rand(), np.pi * 2 * np.random.rand()
        x, y, z = r * np.cos(a), r * np.sin(a) * np.cos(b), r * np.sin(a) * np.sin(b)
        self.goal = self._get_endpoint_position() + np.asarray([x, y, z])

        self.tick = 0

        print self.goal, r, self._check_goal()
        return np.concatenate((self._get_state(), self.goal), 0)

    def step(self, action):
        self.tick += 1
        # print action, self.action_map[action] + self._get_state()
        g = self.action_map[action] + self._get_state()
        v = dict(zip(self.joint_names, self.action_map[action]))
        c = dict(zip(self.joint_names, g))
        # self.arm.move_to_joint_positions(c, 0.5)
        # self.arm.set_joint_positions(c)

        t0 = rospy.get_time()
        while rospy.get_time() - t0 < self.step_dt:
            self.arm.set_joint_positions(c)
            # self.arm.set_joint_velocities(v)

        # check termination conditions
        reached_goal = self._check_goal()
        done = reached_goal or self.tick == self.spec['timestep_limit']
        if done:
            if reached_goal:
                # blink outer LED to signal success
                self.outer_led.set_output(True, 0.)
            else:
                # blink inner LED to signal failure
                self.inner_led.set_output(True, 0.)
        else:
            self.outer_led.set_output(False, 0.)
            self.inner_led.set_output(False, 0.)

        return np.concatenate((self._get_state(), self.goal), 0), -1., done

    def render(self):
        # print self.arm.joint_angles()
        print self._get_endpoint_position(), distance(self.goal, self._get_endpoint_position()), self._check_goal()

def get_baxter_env(env_id):
    parts = env_id.split('.')
    if parts[0] == 'ik':
        if parts[-1] == 'left':
            return BaxterIkEnv(limb='left')
        return BaxterIkEnv(limb='right')


if __name__ == '__main__':
    from core import test_env
    import sys

    env = BaxterIkEnv(sys.argv[1])
    test_env(env, False)
