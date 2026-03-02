"""
Operational Space Controller (OSC) for a 7-DOF arm.

Implements the control law:

    Task-space PD + feedforward:
        xddot_cmd = xddot_des + Kp @ (x_des - x) + Kd @ (xdot_des - xdot)

    Task-space force (with dynamics compensation):
        Λ  = (J M⁻¹ Jᵀ)⁻¹                       task-space inertia
        μ  = Λ (J M⁻¹ h - Jdot qdot)             task-space bias
        F  = Λ xddot_cmd + μ                      commanded EE force
        τ_task = Jᵀ F

    Null-space posture control:
        N  = I - Jᵀ Λ J M⁻¹                      dyn. consistent null-space projector
        τ_null = Nᵀ (Kp_null (q_rest - q) - Kd_null qdot)
        τ  = τ_task + τ_null

References:
    Khatib (1987) "A Unified Approach for Motion and Force Control of Robot
    Manipulators: The Operational Space Formulation"
"""

import numpy as np


class OSC:
    def __init__(
        self,
        Kp=np.diag([200.0, 200.0, 200.0]),
        Kd=np.diag([30.0, 30.0, 30.0]),
        Kp_null=10.0,
        Kd_null=4.0,
        q_rest=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    ):
        """
        Args:
            Kp:      (3,3) task-space position stiffness
            Kd:      (3,3) task-space damping
            Kp_null: scalar null-space posture stiffness
            Kd_null: scalar null-space posture damping
            q_rest:  (7,) desired null-space posture (home config)
        """
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_null = Kp_null
        self.Kd_null = Kd_null
        self.q_rest = q_rest

    def compute(self, x, xdot, x_des, xdot_des, xddot_des,
                q, qdot, M, h, J, Jdot_qdot):
        """
        Compute the joint torques for one OSC tick.

        Args:
            x:          (3,)   current EE position
            xdot:       (3,)   current EE velocity
            x_des:      (3,)   desired EE position
            xdot_des:   (3,)   desired EE velocity
            xddot_des:  (3,)   desired EE acceleration (feedforward)
            q:          (7,)   joint positions
            qdot:       (7,)   joint velocities
            M:          (7,7)  mass matrix
            h:          (7,)   bias forces (C*qdot + g)
            J:          (3,7)  linear Jacobian
            Jdot_qdot:  (3,)   dJ/dt * qdot

        Returns:
            tau: (7,) joint torques
        """
        n = len(q)  # 7

        # --- Task-space PD ---
        e = x_des - x
        edot = xdot_des - xdot
        xddot_cmd = xddot_des + self.Kp @ e + self.Kd @ edot

        # --- Task-space dynamics ---
        M_inv = np.linalg.inv(M)                          # (7, 7)
        Lam_inv = J @ M_inv @ J.T                         # (3, 3)
        Lam = np.linalg.inv(Lam_inv)                      # task-space inertia

        # Task-space bias: μ = Λ (J M⁻¹ h - Jdot qdot)
        mu = Lam @ (J @ M_inv @ h - Jdot_qdot)

        # Commanded EE force
        F_cmd = Lam @ xddot_cmd + mu                      # (3,)

        # Map to joint torques
        tau_task = J.T @ F_cmd                             # (7,)

        # --- Null-space posture control ---
        # Dynamically-consistent pseudo-inverse: J_bar = M⁻¹ Jᵀ Λ
        J_bar = M_inv @ J.T @ Lam                         # (7, 3)
        N = np.eye(n) - J.T @ J_bar.T                     # (7, 7) null-space projector

        tau_null_raw = self.Kp_null * (self.q_rest - q) - self.Kd_null * qdot
        tau_null = N @ tau_null_raw                        # (7,)

        tau = tau_task + tau_null
        return tau

    def step(self, env, x_des, xdot_des, xddot_des):
        """
        One full OSC tick: read state, compute torques, apply, step physics.

        Args:
            env:        PandaEnv instance
            x_des:      (3,) desired EE position
            xdot_des:   (3,) desired EE velocity
            xddot_des:  (3,) desired EE acceleration

        Returns:
            x:   (3,) current EE position (after state read, before step)
            err: float, position error norm
        """
        q, qdot = env.get_joint_states()
        x, xdot = env.get_ee_state()
        M, h = env.get_dynamics(q, qdot)
        J = env.get_jacobian(q)
        Jdot_qdot = env.get_jdot_qdot(q, qdot)

        tau = self.compute(
            x, xdot, x_des, xdot_des, xddot_des,
            q, qdot, M, h, J, Jdot_qdot,
        )
        env.apply_torques(tau)
        env.step()

        return x, np.linalg.norm(x_des - x)
