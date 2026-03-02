"""
Spoofed VLA / goal source using PyBullet debug GUI sliders.

Replaces the VLA/policy layer in the control stack. Provides a target
EE position (x, y, z) that the user can adjust in real time via sliders
in the PyBullet GUI window.

Also renders a visual sphere at the target location so you can see
where the arm is trying to go.

Called at 20 Hz by the main loop (every 50 sim ticks).
"""

import numpy as np
import pybullet as p


class SpoofVLA:
    def __init__(self, physics_client, initial_pos=None):
        """
        Args:
            physics_client: PyBullet client ID (needed for GUI calls)
            initial_pos:    (3,) starting slider values [m]
        """
        self.client = physics_client

        if initial_pos is None:
            initial_pos = np.array([0.307, 0.0, 0.485])

        # --- Create sliders for x, y, z target ---
        self.slider_x = p.addUserDebugParameter(
            "target_x", -0.5, 0.8, initial_pos[0],
            physicsClientId=self.client,
        )
        self.slider_y = p.addUserDebugParameter(
            "target_y", -0.5, 0.5, initial_pos[1],
            physicsClientId=self.client,
        )
        self.slider_z = p.addUserDebugParameter(
            "target_z", 0.0, 0.8, initial_pos[2],
            physicsClientId=self.client,
        )

        # --- Visual marker (small red sphere at the target) ---
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.02,
            rgbaColor=[1, 0, 0, 0.7],
            physicsClientId=self.client,
        )
        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=initial_pos.tolist(),
            physicsClientId=self.client,
        )

        self._last_goal = initial_pos.copy()

    def read_goal(self):
        """
        Read the current slider values and update the visual marker.

        Returns:
            goal: (3,) target EE position [m]
        """
        x = p.readUserDebugParameter(self.slider_x, physicsClientId=self.client)
        y = p.readUserDebugParameter(self.slider_y, physicsClientId=self.client)
        z = p.readUserDebugParameter(self.slider_z, physicsClientId=self.client)

        goal = np.array([x, y, z])

        # Update marker position only if goal changed (avoid unnecessary calls)
        if not np.allclose(goal, self._last_goal, atol=1e-4):
            p.resetBasePositionAndOrientation(
                self.marker_id, goal.tolist(), [0, 0, 0, 1],
                physicsClientId=self.client,
            )
            self._last_goal = goal.copy()

        return goal
