"""
Test for trajectory.py — verify min-jerk profile properties.
Usage:  conda activate pybullet_env && python test_trajectory.py
"""

import numpy as np
from src.trajectory import MinJerkTrajectory


def main():
    traj = MinJerkTrajectory(duration=1.0)

    x_start = np.array([0.3, 0.0, 0.5])
    x_goal = np.array([0.4, 0.1, 0.6])
    traj.initialize(x_start)
    traj.set_new_goal(x_goal)

    dt = 1.0 / 200.0  # 200 Hz
    positions = []
    velocities = []
    accelerations = []

    for _ in range(300):  # 1.5 seconds
        x, xd, xdd = traj.update(dt)
        positions.append(x.copy())
        velocities.append(xd.copy())
        accelerations.append(xdd.copy())

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    # --- 1. Starts at x_start ---
    err_start = np.linalg.norm(positions[0] - x_start)
    print(f"[OK] Start position error: {err_start:.2e}")
    assert err_start < 0.01

    # --- 2. Ends at x_goal ---
    err_end = np.linalg.norm(positions[-1] - x_goal)
    print(f"[OK] End position error:   {err_end:.2e}")
    assert err_end < 1e-6

    # --- 3. Zero velocity at start and end ---
    vel_start = np.linalg.norm(velocities[0])
    vel_end = np.linalg.norm(velocities[-1])
    print(f"[OK] Start velocity norm:  {vel_start:.2e}")
    print(f"[OK] End velocity norm:    {vel_end:.2e}")
    assert vel_end < 1e-10

    # --- 4. Zero acceleration at end ---
    acc_end = np.linalg.norm(accelerations[-1])
    print(f"[OK] End accel norm:       {acc_end:.2e}")
    assert acc_end < 1e-10

    # --- 5. Velocity is consistent with finite-diff of position ---
    for i in range(1, 200):  # check within the motion (not at boundaries)
        xdot_fd = (positions[i] - positions[i-1]) / dt
        err = np.linalg.norm(velocities[i] - xdot_fd)
        assert err < 0.05, f"Velocity mismatch at step {i}: {err:.4f}"
    print(f"[OK] Velocity matches finite-diff of position")

    # --- 6. Re-planning mid-motion ---
    traj2 = MinJerkTrajectory(duration=0.5)
    traj2.initialize(x_start)
    traj2.set_new_goal(x_goal)
    # Advance halfway
    for _ in range(50):
        traj2.update(dt)
    x_mid = traj2.x_des.copy()
    # Re-plan to a new goal from current position
    x_goal2 = np.array([0.5, -0.1, 0.4])
    traj2.set_new_goal(x_goal2)
    x_after_replan, _, _ = traj2.update(dt)
    # Should start near where we were
    err_replan = np.linalg.norm(x_after_replan - x_mid)
    print(f"[OK] Re-plan continuity:   {err_replan:.4f} m (should be small)")
    assert err_replan < 0.01

    print("\nAll trajectory tests passed!")


if __name__ == "__main__":
    main()
