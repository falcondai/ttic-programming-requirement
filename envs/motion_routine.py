import sys

import rospy
import actionlib
from std_msgs.msg import Header
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
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import JointTrajectoryPoint
from quaternion import Quaternion as Quat

def convert_joint_to_dict(joint):
    return dict(zip(joint.name, joint.position))

class Trajectory(object):
    def __init__(self, limb):
        ns = 'robot/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.1)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()
        rospy.sleep(0.1)

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

def to_Quat(ros_q):
    return Quat([ros_q.x, ros_q.w, ros_q.z, ros_q.y])

def get_ik_joints_linear(initial_orientation, initial_position, target_orientation, target_position, n_steps, limb):
    ns = "ExternalTools/%s/PositionKinematicsNode/IKService" % limb
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest(seed_mode=SolvePositionIKRequest.SEED_CURRENT)
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')

    x0, y0, z0 = initial_position
    x1, y1, z1 = target_position

    # print initial_orientation
    q0 = initial_orientation
    # print 'q0', q0.as_v_theta()
    q1 = target_orientation
    # print 'q1', q1, q1.as_v_theta()

    # linear interpolate between current pose and target pose
    for i in xrange(n_steps):
        t = (i + 1) * 1. / n_steps
        x = (1. - t) * x0 + t * x1
        y = (1. - t) * y0 + t * y1
        z = (1. - t) * z0 + t * z1

        q = (q1 * q0 ** -1.) ** t * q0
        qw, qx, qy, qz = q.as_wxyz()
        # print q.as_v_theta()

        pose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point( x=x, y=y, z=z, ),
                # endeffector pointing down
                # orientation=Quaternion(x=qw, y=qz, z=qy, w=qx),
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

def get_linear_trajectory_from_current(arm, target_x, target_y, target_z, target_orientation, n_steps, duration):
    limb = arm.name
    traj = Trajectory(limb)
    current_pose = arm.endpoint_pose()
    current_position = current_pose['position']
    dt_step = duration * 1. / n_steps
    js = get_ik_joints_linear(to_Quat(current_pose['orientation']), (current_position.x, current_position.y, current_position.z), target_orientation, (target_x, target_y, target_z), n_steps, limb)
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

def execute_linear(arm, target_x, target_y, target_z, target_orientation, n_steps=10, duration=4., timeout=10.):
    traj = get_linear_trajectory_from_current(arm, target_x, target_y, target_z, target_orientation, n_steps, duration)
    if traj == None:
        return None
    traj.start()
    traj.wait(timeout)
    return traj

def execute_vertical(arm, target_z, n_steps=10, duration=4., timeout=10.):
    pos = arm.endpoint_pose()['position']
    o = arm.endpoint_pose()['orientation']
    q0 = to_Quat(o)
    return execute_linear(arm, pos.x, pos.y, target_z, q0, n_steps, duration, timeout)

def execute_horizontal(arm, target_x, target_y, target_theta, n_steps=10, duration=4., timeout=10.):
    pos = arm.endpoint_pose()['position']
    return execute_linear(arm, target_x, target_y, pos.z, Quat.from_v_theta([0, 0, 1], target_theta), n_steps, duration, timeout)

def execute_planar_grasp(arm, gripper, initial_z, target_z, n_steps=10, duration=4., timeout=10., sleep=1., lower_to_drop=0.):
    traj = execute_vertical(arm, target_z, n_steps, duration, timeout)
    if traj.result().error_code == 0:
        gripper.close()
        rospy.sleep(sleep)
        execute_vertical(arm, initial_z, n_steps, duration, timeout)
        if lower_to_drop > 0:
            execute_vertical(arm, target_z + lower_to_drop, n_steps, duration, timeout)
    gripper.open()
    rospy.sleep(sleep)

if __name__ == '__main__':
    import baxter_interface
    import numpy as np

    rospy.init_node('grasp')
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    rs.enable()

    x0 = 0.81
    y0 = 0.25
    delta = 0.04
    initial_z = 0.1
    bound_z = -0.165

    arm = baxter_interface.Limb('left')
    gripper = baxter_interface.Gripper('left')
    gripper.calibrate()

    for _ in xrange(4):
        execute_linear(arm, x0, y0, initial_z, Quat.from_v_theta([0,0,1], np.random.rand() * 2 * np.pi), n_steps=1, duration=5.)
        rospy.sleep(1.)
        execute_planar_grasp(arm, gripper, initial_z, bound_z, n_steps=1)
