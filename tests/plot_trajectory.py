"""Plot the min-jerk trajectory profile for x_des, xdot_des, xddot_des."""

import numpy as np
import matplotlib.pyplot as plt
from src.trajectory import MinJerkTrajectory


traj = MinJerkTrajectory(duration=1.0)
x_start = np.array([0.3, 0.0, 0.5])
x_goal = np.array([0.4, 0.1, 0.6])
traj.initialize(x_start)
traj.set_new_goal(x_goal)

dt = 1.0 / 200.0
times, pos, vel, acc = [], [], [], []

for i in range(300):  # 1.5s
    x, xd, xdd = traj.update(dt)
    times.append(i * dt)
    pos.append(x.copy())
    vel.append(xd.copy())
    acc.append(xdd.copy())

times = np.array(times)
pos = np.array(pos)
vel = np.array(vel)
acc = np.array(acc)
labels = ["x", "y", "z"]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(3):
    axes[0].plot(times, pos[:, i], label=labels[i])
    axes[1].plot(times, vel[:, i], label=labels[i])
    axes[2].plot(times, acc[:, i], label=labels[i])

axes[0].set_ylabel("x_des [m]")
axes[0].set_title("Minimum-Jerk Trajectory Profile")
axes[0].legend()
axes[0].axvline(1.0, color="k", ls="--", alpha=0.3, label="T=1s")
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("xdot_des [m/s]")
axes[1].axvline(1.0, color="k", ls="--", alpha=0.3)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].set_ylabel("xddot_des [m/s²]")
axes[2].set_xlabel("time [s]")
axes[2].axvline(1.0, color="k", ls="--", alpha=0.3)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("tests/trajectory_profile.png", dpi=150)
plt.show()
print("Saved to tests/trajectory_profile.png")
