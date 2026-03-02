# osc_control

Operational Space Controller (OSC) for a 7-DOF Franka Panda arm in PyBullet.

<video src="https://github.com/shubhamsingh91/osc_control/raw/main/assets/demo.mp4" width="640" autoplay loop muted></video>

## Overview

A from-scratch implementation of Khatib's Operational Space formulation (1987) with:

- **Task-space PD control** with feedforward acceleration
- **Damped pseudo-inverse** for singularity robustness
- **Null-space posture control** to prevent elbow drift
- **Minimum-jerk trajectory generator** for smooth reference tracking
- **Multi-rate control loop** mirroring a real robot stack:
  - 1000 Hz — OSC torque computation + physics
  - 200 Hz — Trajectory reference updates
  - 20 Hz — Goal source (spoofed VLA via PyBullet sliders)

Dynamics (mass matrix, Coriolis, Jacobians) are computed via **Pinocchio**.

## Control Law

```
xddot_cmd = xddot_des + Kp (x_des - x) + Kd (xdot_des - xdot)

Lambda = (J M^-1 J^T + lambda I)^-1       # damped task-space inertia
mu     = Lambda (J M^-1 h - Jdot qdot)    # task-space bias
F      = Lambda xddot_cmd + mu            # EE force command
tau    = J^T F + N^T (Kp_null (q_rest - q) - Kd_null qdot)
```

## Project Structure

```
osc_control/
├── main.py                 # Multi-rate main loop (run this)
├── src/
│   ├── env.py              # PyBullet sim + Pinocchio dynamics
│   ├── osc.py              # OSC math (task-space PD, null-space)
│   ├── trajectory.py       # Minimum-jerk trajectory generator
│   └── spoof_vla.py        # PyBullet GUI sliders for target position
├── tests/
│   ├── test_env.py         # Env wrapper tests (dynamics, Jacobian)
│   ├── test_osc.py         # OSC convergence test
│   ├── test_trajectory.py  # Trajectory profile tests
│   └── plot_trajectory.py  # Visualize min-jerk profiles
└── assets/
    └── demo.gif
```

## Setup

Requires a conda environment with Pinocchio, PyBullet, NumPy, and SciPy:

```bash
conda create -n pybullet_env python=3.9
conda activate pybullet_env
conda install -c conda-forge pinocchio
pip install pybullet numpy scipy
```

## Usage

### Run the controller

```bash
conda activate pybullet_env
python main.py
```

A PyBullet GUI window opens with the Panda arm and three sliders (target_x, target_y, target_z). Move the sliders to command the end-effector to a new position. A red sphere shows the current target.

### Run tests

```bash
conda activate pybullet_env
python tests/test_env.py
python tests/test_osc.py
python tests/test_trajectory.py
```

## References

- O. Khatib, "A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation," IEEE Journal of Robotics and Automation, 1987.
