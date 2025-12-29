"""Take in joint velocity angles into JointState msg for ROS 2."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco

import numpy as np
import os
import sys
from mpc import Mpc


current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)


class Sim(Node):
    """Create Simulator to connect controller with ROS."""

    def __init__(self):
        """Initialize variables."""
        super.__init__('sim')
        self._pub = self.create_publisher(JointState, 'joint_states', 10)
        model_path = os.path.join(current_dir, 'franka_emika_panda/panda.xml')

        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self.model)
        self._mpc = Mpc(self.model, self.data)
        self._target = np.array([0.5, 0.0, 0.5])
        self._obs = np.array([0.3, 0.0, 0.3])

        # Set Ready Pose
        self._data.qpos[:9] = [0.0,
                               0.7853981633974483,
                               0.0,
                               -2.356194490192345,
                               0.0,
                               1.5707963267948966,
                               0.7853981633974483,
                               ]
        self._target = np.array([0.5, 0.0, 0.5])
        self._obs = np.array([0.5, 0.0, 0.3])
        self._timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        """Call callback function for timer."""
        q = self._data.qpos[:7]
        q_dot = self._mpc.calculate_joint_vel(q, self._target, self._obs)
        self._data.qpos[:7] += q_dot * 0.01
        mujoco.mj_kinematics(self._model, self._data)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3',
                    'panda_joint4', 'panda_joint5', 'panda_joint6',
                    'panda_joint7', 'panda_finger_joint1',
                    'panda_finger_joint2']
        msg.position = self._data.qpos[:9].tolist()
        self._pub.publish(msg)


def main():
    """Spin node."""
    rclpy.init()
    node = Sim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()