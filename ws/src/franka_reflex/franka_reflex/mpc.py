"""Customized Controller for obstacle avoidance."""
import numpy as np
import mujoco


class Mpc():
    """Controller for motion planning."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Initialize all parameters."""
        self.model = model
        self.data = data
        self.collision_bodies = [
            'link1', 'link2', 'link3', 'link4',
            'link5', 'link6', 'link7', 'hand',
            'left_finger', 'right_finger'
        ]
        self.bodies = []
        for collision in self.collision_bodies:
            body = mujoco.mj_name2id(model,
                                     mujoco.mjtObj.mjOBJ_BODY,
                                     collision)
            self.bodies.append(body)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

        self.threshold = 0.2  # Safety threshold from obstacle
        self.zeta = 40.0  # Attraction gain
        self.zeta_ori = 2.0
        self.eta = 10.0  # Repulsion gain
        self.k_pose = 0.05
        self.ready_pos = np.array([0.0,
                                   -0.7853981633974483,
                                   0.0,
                                   -2.356194490192345,
                                   0.0,
                                   1.5707963267948966,
                                   0.7853981633974483])
        self.target_rot = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])

    def calculate_joint_vel(self,
                            q: np.ndarray,
                            target_pos: np.array,
                            obs_pos: np.ndarray) -> np.array:
        """
        Calculate output joint velocity.

        :param q: current joint configuration
        :param target_pos: target cartesian postion [x,y,z]
        :param obs_pos: obstacle cartesian position [x,y,z]
        :return: joint velocity
        """
        total_forces = np.zeros(7)

        # Find the ee position offsetted from the hand
        hand_pos = self.data.xpos[self.ee_id]
        hand_rot = self.data.xmat[self.ee_id].reshape(3, 3)
        offset = np.array([0.0, 0.0, 0.105])
        ee_pos = hand_pos + hand_rot @ offset

        for body in self.bodies:
            current_pos = self.data.xpos[body]
            f_rep = self.calculate_repulsion(current_pos, obs_pos)
            J = self.calculate_jacobian(body)
            qdot = J.T @ f_rep
            total_forces += qdot

        # Attraction Force
        error_pos = target_pos - ee_pos
        f_att = self.zeta * error_pos
        error_ori = self.calculate_orientation(hand_rot, self.target_rot)
        f_att_ori = self.zeta_ori * error_ori
        F = np.hstack([f_att, f_att_ori])
        J_6D = np.zeros((6, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_6D[:3], J_6D[3:], self.ee_id)
        J_6D = J_6D[:, :7]
        total_forces += J_6D.T @ F

        dist_to_goal = np.linalg.norm(target_pos - ee_pos)
        qdot_posture = self.k_pose * (self.ready_pos - q)
        qdot_total = total_forces + qdot_posture
        qdot_total *= 0.8

        if dist_to_goal < 0.05:
            speed_scale = dist_to_goal / 0.05
            qdot_total *= speed_scale

        if dist_to_goal < 0.005:
            return np.zeros(7)

        return np.clip(qdot_total, -1.0, 1.0)

    def calculate_repulsion(self,
                            current_pos: np.ndarray,
                            obs_pos: np.ndarray):
        """
        Calculate repulsive force for motion planning.

        :param current_pos: current cartesian position of the ee
        :param obs_pos: obstacle cartesian position
        """
        # Calculate repulsion like magenetic field
        # that pushes robot away from obstacle
        d = current_pos - obs_pos
        d_scalar = np.linalg.norm(d)
        f_rep = np.zeros(3)

        if d_scalar < self.threshold:
            f = self.eta*(1/d_scalar - 1/self.threshold) * 1/(d_scalar**2)
            if f > 50.0:
                f = 50.0
            direc = d/d_scalar
            f_rep = f*direc
        return f_rep

    def calculate_jacobian(self, body):
        """Calculate Jacobian to transform force into joint torque."""
        j_init = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, j_init, None, body)
        jacobian = j_init[:, :7]
        return jacobian
    
    def calculate_orientation(self, robot_rot, target_rot):
        """
        Calculate the rotation vector to align robot hand with target.
        
        :param robot_rot: Rotation vector of the robot hand
        :param target_rot: Rotation vector of the target
        """
        r_x = robot_rot[:, 0]
        r_y = robot_rot[:, 1]
        r_z = robot_rot[:, 2]
        t_x = target_rot[:, 0]
        t_y = target_rot[:, 1]
        t_z = target_rot[:, 2]

        e_x = np.cross(r_x, t_x)
        e_y = np.cross(r_y, t_y)
        e_z = np.cross(r_z, t_z)

        return 0.5*(e_x + e_y + e_z)