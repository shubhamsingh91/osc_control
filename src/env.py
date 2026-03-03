"""
PyBullet environment wrapper for the Franka Panda arm.

Responsibilities:
  - PyBullet: physics simulation, state readout, torque application
  - Pinocchio: dynamics (M, h) and kinematics (J) — more accurate than
    PyBullet's built-in dynamics functions
"""

import os

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data


# Panda arm: joints 0-6 are the 7 revolute DOFs
ARM_JOINT_INDICES = list(range(7))
NUM_ARM_JOINTS = 7
NUM_ALL_MOVABLE = 9  # 7 arm + 2 finger (pinocchio model has nq=9)
# End-effector link index in PyBullet (panda_grasptarget)
EE_LINK_INDEX = 11
# EE frame name in Pinocchio
EE_FRAME_NAME = "panda_grasptarget"
CAMERA_PRESETS = {
    "front": dict(distance=1.10, yaw=45.0, pitch=-30.0, roll=0.0),
    "side": dict(distance=1.05, yaw=110.0, pitch=-25.0, roll=0.0),
    "top": dict(distance=1.20, yaw=0.0, pitch=-80.0, roll=0.0),
}


class PandaEnv:
    def __init__(self, dt=1.0 / 1000.0, gui=True):
        """
        Args:
            dt:  simulation timestep in seconds (default 1 ms → 1 kHz physics)
            gui: if True, open a PyBullet GUI window; else use DIRECT mode
        """
        self.dt = dt

        # --- Connect to PyBullet ---
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # --- Physics parameters ---
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(dt, physicsClientId=self.client)
        # Disable default velocity/position motor so we have pure torque control
        # (done per-joint below after loading the robot)

        # --- Load ground plane + robot ---
        self.plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self.client
        )
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client,
        )

        # --- Put arm joints into torque-control mode ---
        # PyBullet defaults joints to VELOCITY_CONTROL with some damping.
        # We must disable that and switch to TORQUE_CONTROL.
        for j in ARM_JOINT_INDICES:
            p.setJointMotorControl2(
                self.robot_id,
                j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,  # zero out the default motor
                physicsClientId=self.client,
            )

        # --- Set a nice initial configuration (roughly "home" pose) ---
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        for i, j in enumerate(ARM_JOINT_INDICES):
            p.resetJointState(self.robot_id, j, self.q_home[i],
                              physicsClientId=self.client)

        # --- Read joint limits for reference ---
        self.joint_lower = np.zeros(NUM_ARM_JOINTS)
        self.joint_upper = np.zeros(NUM_ARM_JOINTS)
        self.joint_max_force = np.zeros(NUM_ARM_JOINTS)
        for i, j in enumerate(ARM_JOINT_INDICES):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            self.joint_lower[i] = info[8]
            self.joint_upper[i] = info[9]
            self.joint_max_force[i] = info[10]

        # --- Pinocchio model (for dynamics & kinematics) ---
        urdf_path = os.path.join(
            pybullet_data.getDataPath(), "franka_panda", "panda.urdf"
        )
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        self.ee_frame_id = self.pin_model.getFrameId(EE_FRAME_NAME)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_joint_states(self):
        """
        Returns:
            q:    (7,) joint positions  [rad]
            qdot: (7,) joint velocities [rad/s]
        """
        states = p.getJointStates(
            self.robot_id, ARM_JOINT_INDICES, physicsClientId=self.client
        )
        q = np.array([s[0] for s in states])
        qdot = np.array([s[1] for s in states])
        return q, qdot

    def get_ee_state(self):
        """
        Returns:
            pos:  (3,) EE position in world frame [m]
            vel:  (3,) EE linear velocity in world frame [m/s]

        (Orientation and angular velocity omitted for now —
         we're doing position-only OSC first.)
        """
        link_state = p.getLinkState(
            self.robot_id,
            EE_LINK_INDEX,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos = np.array(link_state[4])   # worldLinkFramePosition
        vel = np.array(link_state[6])   # worldLinkLinearVelocity
        return pos, vel

    def get_camera_rgb(self, view="front", width=256, height=256):
        """
        Render a deterministic RGB image from a fixed camera viewpoint.

        Args:
            view: one of {"front", "side", "top"}
            width: output image width in pixels
            height: output image height in pixels

        Returns:
            rgb: (height, width, 3) uint8 image
        """
        if view not in CAMERA_PRESETS:
            raise ValueError(f"Unknown camera view '{view}'.")

        cfg = CAMERA_PRESETS[view]
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.35, 0.0, 0.35],
            distance=cfg["distance"],
            yaw=cfg["yaw"],
            pitch=cfg["pitch"],
            roll=cfg["roll"],
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=3.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3]

    # ------------------------------------------------------------------
    # Dynamics & Kinematics  (via Pinocchio)
    # ------------------------------------------------------------------

    def _pad_q(self, q):
        """Pad 7-DOF arm q to 9-DOF (append finger joints as zeros)."""
        q_full = np.zeros(NUM_ALL_MOVABLE)
        q_full[:NUM_ARM_JOINTS] = q
        return q_full

    def get_dynamics(self, q, qdot):
        """
        Compute joint-space dynamics using Pinocchio's CRBA and NLE.

        Args:
            q:    (7,) arm joint positions
            qdot: (7,) arm joint velocities

        Returns:
            M: (7, 7) mass/inertia matrix (arm block)
            h: (7,)   bias forces  =  C(q, qdot)*qdot + g(q)
        """
        q_full = self._pad_q(q)
        v_full = np.zeros(NUM_ALL_MOVABLE)
        v_full[:NUM_ARM_JOINTS] = qdot

        # M(q) via Composite Rigid Body Algorithm
        M_full = pin.crba(self.pin_model, self.pin_data, q_full)
        M = np.array(M_full[:NUM_ARM_JOINTS, :NUM_ARM_JOINTS])

        # h(q, qdot) = C(q,qdot)*qdot + g(q)  via nonLinearEffects
        h_full = pin.nonLinearEffects(self.pin_model, self.pin_data, q_full, v_full)
        h = np.array(h_full[:NUM_ARM_JOINTS])

        return M, h

    def get_jacobian(self, q):
        """
        Compute the 3×7 linear Jacobian of the EE frame using Pinocchio.

        Args:
            q: (7,) arm joint positions

        Returns:
            J: (3, 7) linear Jacobian  dx/dq  (world frame)
        """
        q_full = self._pad_q(q)

        # pin.LOCAL_WORLD_ALIGNED = expressed in world frame, at the frame origin
        pin.computeJointJacobians(self.pin_model, self.pin_data, q_full)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J_full = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED
        )
        # J_full is (6, 9): rows 0-2 = linear, rows 3-5 = angular
        J = J_full[:3, :NUM_ARM_JOINTS]  # (3, 7) linear part, arm only
        return J

    def get_jdot_qdot(self, q, qdot):
        """
        Compute Jdot * qdot  (3,) for the EE frame using Pinocchio.

        This is the "drift" acceleration of the EE due to joint velocities
        even when joint accelerations are zero. Needed by the OSC to cancel
        the nonlinear velocity-dependent term.

        Args:
            q:    (7,) arm joint positions
            qdot: (7,) arm joint velocities

        Returns:
            Jdot_qdot: (3,) = dJ/dt @ qdot  (linear part, world frame)
        """
        q_full = self._pad_q(q)
        v_full = np.zeros(NUM_ALL_MOVABLE)
        v_full[:NUM_ARM_JOINTS] = qdot

        # computeJointJacobiansTimeVariation needs FK + Jacobians first
        pin.computeJointJacobians(self.pin_model, self.pin_data, q_full)
        pin.computeJointJacobiansTimeVariation(
            self.pin_model, self.pin_data, q_full, v_full
        )
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        dJ_full = pin.getFrameJacobianTimeVariation(
            self.pin_model, self.pin_data, self.ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED
        )
        # dJ_full is (6, 9), we want linear rows (0:3) dotted with v
        Jdot_qdot = dJ_full[:3, :NUM_ARM_JOINTS] @ qdot  # (3,)
        return Jdot_qdot

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def apply_torques(self, tau):
        """
        Apply joint torques to the 7 arm joints.

        Args:
            tau: (7,) desired torques [Nm]
        """
        # Clip to max allowable torque per joint
        tau_clipped = np.clip(tau, -self.joint_max_force, self.joint_max_force)
        p.setJointMotorControlArray(
            self.robot_id,
            ARM_JOINT_INDICES,
            controlMode=p.TORQUE_CONTROL,
            forces=tau_clipped.tolist(),
            physicsClientId=self.client,
        )

    def step(self):
        """Advance the simulation by one timestep."""
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        """Disconnect from the physics server."""
        p.disconnect(physicsClientId=self.client)
