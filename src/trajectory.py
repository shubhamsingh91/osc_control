"""
Minimum-jerk trajectory generator.

Generates smooth references x_des(t), xdot_des(t), xddot_des(t) between
a start position and a goal position using the minimum-jerk profile:

    s(t) = 10(t/T)^3 - 15(t/T)^4 + 6(t/T)^5

This gives zero velocity and acceleration at both endpoints, and minimizes
the integral of jerk squared — the smoothest possible point-to-point motion.

Called at 200 Hz by the main loop (every 5 sim ticks).
Re-plans from the current state whenever a new goal arrives.
"""

import numpy as np


class MinJerkTrajectory:
    def __init__(self, duration=1.0):
        """
        Args:
            duration: default motion duration in seconds for each new goal
        """
        self.duration = duration

        # Current trajectory state
        self.x_start = None     # (3,) start position
        self.x_goal = None      # (3,) goal position
        self.t = 0.0            # elapsed time since trajectory start
        self.T = duration       # total duration for current segment

        # Outputs (held between updates)
        self.x_des = np.zeros(3)
        self.xdot_des = np.zeros(3)
        self.xddot_des = np.zeros(3)

    def initialize(self, x_current):
        """Set the initial position (call once at startup)."""
        self.x_start = x_current.copy()
        self.x_goal = x_current.copy()
        self.x_des = x_current.copy()
        self.t = 0.0
        self.T = self.duration

    def set_new_goal(self, x_goal, x_current=None, duration=None):
        """
        Set a new goal and re-plan from the current trajectory position.

        Args:
            x_goal:    (3,) new target position
            x_current: (3,) current position to start from (if None, uses
                       the current x_des as the start — seamless re-planning)
            duration:  motion duration for this segment (if None, uses default)
        """
        self.x_start = x_current.copy() if x_current is not None else self.x_des.copy()
        self.x_goal = x_goal.copy()
        self.t = 0.0
        self.T = duration if duration is not None else self.duration

    def update(self, dt):
        """
        Advance the trajectory by dt seconds and return the current refs.

        Args:
            dt: time step (e.g., 1/200 for 200 Hz)

        Returns:
            x_des:      (3,) desired position
            xdot_des:   (3,) desired velocity
            xddot_des:  (3,) desired acceleration
        """
        if self.x_start is None:
            return self.x_des, self.xdot_des, self.xddot_des

        self.t += dt

        if self.t >= self.T:
            # Trajectory complete — hold at goal with zero vel/accel
            self.x_des = self.x_goal.copy()
            self.xdot_des = np.zeros(3)
            self.xddot_des = np.zeros(3)
        else:
            # Minimum-jerk interpolation
            tau = self.t / self.T                  # normalized time [0, 1]
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau3 * tau
            tau5 = tau4 * tau

            # Position:  s = 10τ³ - 15τ⁴ + 6τ⁵
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5

            # Velocity:  ds/dt = (1/T)(30τ² - 60τ³ + 30τ⁴)
            sdot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / self.T

            # Acceleration: d²s/dt² = (1/T²)(60τ - 180τ² + 120τ³)
            sddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (self.T * self.T)

            dx = self.x_goal - self.x_start
            self.x_des = self.x_start + s * dx
            self.xdot_des = sdot * dx
            self.xddot_des = sddot * dx

        return self.x_des.copy(), self.xdot_des.copy(), self.xddot_des.copy()
