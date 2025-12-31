"""Take in joint velocity angles into JointState msg for ROS 2."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
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
        super().__init__('sim')
        self._pub = self.create_publisher(JointState, 'joint_states', 10)
        self.marker_pub = self.create_publisher(Marker,
                                                'visualization_marker',
                                                10)
        model_path = os.path.join(current_dir, 'franka_emika_panda/panda.xml')

        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)

        # Set Ready Pose
        self._data.qpos[:9] = [0.0,
                               -0.7853981633974483,
                               0.0,
                               -2.356194490192345,
                               0.0,
                               1.5707963267948966,
                               0.7853981633974483,
                               0.0, 0.0]
        mujoco.mj_forward(self._model, self._data)

        self._mpc = Mpc(self._model, self._data)
        self._target = np.array([0.8, 0.2, 0.0])
        self._obs = np.array([0.5, 0.0, 0.5])
        self._timer = self.create_timer(0.01, self.timer_callback)

    def sphere_marker(self,
                      center: np.ndarray,
                      color: list,
                      scale: float,
                      marker_id: int) -> None:
        """
        Create a spherical marker for obstacle, start and target ee pos.

        :param center: center cartesian position [x,y,z] of the sphere
        :param color: color rgb for the sphere marker [r,g,b]
        :param scale: diameter of the sphere
        :param marker_id: marker id to not confuse between different markers
        """
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = marker_id
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.color.a = 1.0
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        self.marker_pub.publish(marker)

    def path_marker(self):
        """
        Create a marker for tracking the path for ee pose.

        :param self: Description
        """
        return

    def timer_callback(self):
        """Call callback function for timer."""
        q = self._data.qpos[:7]
        q_dot = self._mpc.calculate_joint_vel(q, self._target, self._obs)
        self._data.qpos[:7] += q_dot * 0.01
        mujoco.mj_kinematics(self._model, self._data)
        # --- Debugging Message-------------------------------------
        current_pos = self._data.xpos[9]
        dist = np.linalg.norm(self._target - current_pos)
        print(f'Dist to Goal: {dist:.4f} | Hand: {current_pos}')
        # ----------------------------------------------------------
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3',
                    'panda_joint4', 'panda_joint5', 'panda_joint6',
                    'panda_joint7', 'panda_finger_joint1',
                    'panda_finger_joint2']
        msg.position = self._data.qpos[:9].tolist()
        self._pub.publish(msg)

        # Publihs marker fro obstacle in red
        self.sphere_marker(self._obs,
                           color=[1.0, 0.0, 0.0],
                           scale=0.2,
                           marker_id=0)
        # Publihs marker for target in green
        self.sphere_marker(self._target,
                           color=[0.0, 1.0, 0.0],
                           scale=0.025,
                           marker_id=1)


def main():
    """Spin node."""
    rclpy.init()
    node = Sim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
