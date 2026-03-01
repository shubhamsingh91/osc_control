# OSC Controller for Franka Panda in PyBullet

## Context
Build an Operational Space Controller (OSC) from scratch for a 7-DOF Franka Panda arm simulated in PyBullet. No VLA — goals will be spoofed via PyBullet debug sliders. User wants to understand each piece, so we proceed step-by-step with approval between each step.

## Dependencies
- `uv` for isolated env management
- `pybullet`, `numpy`, `scipy`

## File Structure
```
osc_control/
├── pyproject.toml
├── env.py            # Step 1: PyBullet env wrapper (load URDF, step sim, read state)
├── osc.py            # Step 2: Core OSC math (task-space inertia, torque computation)
├── trajectory.py     # Step 3: Minimum-jerk trajectory generator (smooth refs)
├── spoof_vla.py      # Step 4: Spoofed goal source (PyBullet sliders for x/y/z)
├── main.py           # Step 5: Main loop wiring everything together
```

## Steps (one at a time, user reviews before next)

### Step 1 — `env.py`: PyBullet Environment
- Load `franka_panda/panda.urdf` from pybullet_data
- Torque control mode on all 7 arm joints (not grippers)
- Functions: `get_joint_states()` → q, qdot
- Functions: `get_ee_state()` → pos, orn, linear_vel, angular_vel
- Functions: `get_dynamics(q, qdot)` → M(q), h(q,qdot) via `calculateMassMatrix` + `calculateInverseDynamics`
- Functions: `get_jacobian(q, qdot)` → J (6×7) via `calculateJacobian`
- Functions: `apply_torques(tau)`, `step()`

### Step 2 — `osc.py`: OSC Core
The actual control law each tick:
```
e = x_des - x
edot = xdot_des - xdot
xddot_cmd = xddot_des + Kp @ e + Kd @ edot

Λ = (J @ M⁻¹ @ Jᵀ)⁻¹           # task-space inertia
μ = Λ @ (J @ M⁻¹ @ h)            # task-space Coriolis+gravity
F_cmd = Λ @ xddot_cmd + μ        # (Jdot*qdot ≈ 0 at moderate speeds)
τ_task = Jᵀ @ F_cmd

# Null-space posture control
N = I - J⁺ @ J                    # null-space projector (dynamically consistent)
τ_null = Nᵀ @ (Kp_null*(q_rest - q) - Kd_null*qdot)
τ = τ_task + τ_null
```
- Position-only (3D) first, orientation later if desired
- Tunable Kp, Kd gains

### Step 3 — `trajectory.py`: Reference Generator
- Minimum-jerk interpolation from current x to x_goal
- Outputs: x_des(t), xdot_des(t), xddot_des(t)
- Re-plans when a new goal arrives

### Step 4 — `spoof_vla.py`: Fake Goal Source
- PyBullet `addUserDebugParameter` sliders for target x, y, z
- Visual sphere marker at the target location
- Reads slider values each "VLA tick" (e.g., every 0.5s or on change)

### Step 5 — `main.py`: Multi-Rate Main Loop
- Sim timestep: **1/1000s** (1 ms) → each `stepSimulation()` = one OSC tick
- Three rates, all driven by a tick counter:

```
dt_sim  = 1/1000   # 1 kHz  — physics + OSC (every tick)
dt_traj = 1/200    # 200 Hz — trajectory ref update (every 5 ticks)
dt_vla  = 1/20     # 20 Hz  — read goal sliders (every 50 ticks)

tick = 0
while running:
    # --- 20 Hz: VLA layer ---
    if tick % 50 == 0:
        goal = spoof_vla.read_goal()        # read sliders
        trajectory.set_new_goal(goal)

    # --- 200 Hz: Trajectory generator ---
    if tick % 5 == 0:
        x_des, xdot_des, xddot_des = trajectory.update(dt_traj)

    # --- 1000 Hz: OSC + physics ---
    q, qdot = env.get_joint_states()
    x, xdot = env.get_ee_state()
    M, h     = env.get_dynamics(q, qdot)
    J        = env.get_jacobian(q, qdot)
    tau      = osc.compute(x, xdot, x_des, xdot_des, xddot_des,
                           q, qdot, M, h, J)
    env.apply_torques(tau)
    env.step()
    tick += 1
```

This mirrors the real control stack: slow perception → medium trajectory smoothing → fast torque servo.

## Verification
- Run `uv run main.py`
- Move sliders → arm should track the target sphere smoothly
- Arm should hold posture in null-space (elbow doesn't drift wildly)
- Print loop timing to confirm rates are maintained
