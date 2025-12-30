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

        self.threshold = 0.15  # Safety threshold from obstacle
        self.zeta = 10.0  # Attraction gain
        self.eta = 20.0  # Repulsion gain
        self.k_pose = 2.0
        self.ready_pos = [0.0,
                          -0.7853981633974483,
                          0.0,
                          -2.356194490192345,
                          0.0,
                          1.5707963267948966,
                          0.7853981633974483]

    def calculate_joint_vel(self,
                            q: np.ndarray,
                            target_pos: np.ndarray,
                            obs_pos: np.ndarray) -> np.ndarray:
        """
        Calculate output joint velocity.

        :param q: current joint configuration
        :param target_pos: target cartesian postion [x,y,z]
        :param obs_pos: obstacle cartesian position [x,y,z]
        :return: joint velocity
        """
        total_forces = np.zeros(7)
        for body in self.bodies:
            current_pos = self.data.xpos[body]
            f_rep = self.calculate_repulsion(current_pos, obs_pos)
            f_att = np.zeros(3)
            if body == self.ee_id:
                error = target_pos - current_pos
                f_att = self.zeta * error
            F = f_att + f_rep
            J = self.calculate_jacobian(body)
            qdot = J.T @ F
            total_forces += qdot

        qdot_posture = self.k_pose * (self.ready_pos - q)
        qdot_total = total_forces + qdot_posture

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
            direc = d/d_scalar
            f_rep = f*direc
        return f_rep

    def calculate_jacobian(self, body):
        """Calculate Jacobian to transform force into joint torque."""
        j_init = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, j_init, None, body)
        jacobian = j_init[:, :7]
        return jacobian
