"""Customized Controller for obstacle avoidance."""
import numpy as np
import mujoco


class Mpc():
    """Controller for motion planning."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Initialize all parameters."""
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        
        self.threshold = 0.2  # Safety threshold from obstacle
        self.zeta = 5.0  # Attraction gain
        self.eta = 10.0  # Repulsion gain
        self.k_pose = 2.0
        self.ready_pos = [0.0,
                          0.7853981633974483,
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
        current_pos = self.data.xpos[self.ee_id]
        F = self.calculate_potential_force(current_pos, target_pos, obs_pos)
        J = self.calculate_jacobian()
        qdot = J.T @ F
        qdot_posture = self.k_pose * (self.ready_pos - q)
        qdot_total = qdot + qdot_posture
        return np.clip(qdot_total, -0.05, 0.05)
    
    def calculate_potential_force(self,
                                  current_pos: np.ndarray,
                                  target_pos: np.ndarray,
                                  obs_pos: np.ndarray):
        """
        Calculate potential force for motion planning.
        
        :param current_pos: current cartesian position of the ee
        :param target_pos: target cartesian position of ee
        :param obs_pos: obstacle cartesian position
        """
        # Calculate attraction like a spring that pulls robot toward the goal
        error = target_pos - current_pos
        f_att = self.zeta*error

        # Calculate repulsion like magenetic field
        # that pushes robot away from obstacle
        d = current_pos - obs_pos
        d_scalar = np.linalg.norm(d)
        f_rep = np.zeros(3)

        if d_scalar < self.threshold:
            f = self.eta*(1/d_scalar - 1/self.threshold) * 1/(d_scalar**2)
            direc = d/d_scalar
            f_rep = f*direc

        # Find total potential force by adding attraction and repulsion
        f_total = f_att + f_rep
        return f_total
    
    def calculate_jacobian(self):
        """Calculate Jacobian to transform force into joint torque."""
        j_init = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, j_init, None, self.ee_id)
        jacobian = j_init[:, :7]
        return jacobian
