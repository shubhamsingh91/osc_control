"""
Multi-rate main loop for the OSC controller.

Mirrors a real robot control stack with three rates:
    - 5 Hz:    VLA / goal source
    - 200 Hz:  Trajectory generator (smooth min-jerk refs)
    - 1000 Hz: OSC torque computation + physics step

Usage:
    source ~/.venvs/osc/bin/activate
    python main.py            # SmolVLA policy
    python main.py --spoof    # GUI slider control
"""

import argparse
import time

import numpy as np

from src.env import PandaEnv
from src.osc import OSC
from src.trajectory import MinJerkTrajectory


# ---------------------------------------------------------------------------
# Rate configuration
# ---------------------------------------------------------------------------
DT_SIM = 1.0 / 1000.0     # 1 kHz — physics + OSC
TICKS_PER_TRAJ = 5         # 200 Hz — trajectory update every 5 ticks
TICKS_PER_VLA = 200        # 5 Hz   — goal read every 200 ticks
DT_TRAJ = DT_SIM * TICKS_PER_TRAJ   # 5 ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spoof", action="store_true",
                        help="Use GUI sliders instead of SmolVLA")
    args = parser.parse_args()

    # --- Initialize all components ---
    env = PandaEnv(dt=DT_SIM, gui=True)
    osc = OSC()
    traj = MinJerkTrajectory(duration=1.0)

    if args.spoof:
        from src.spoof_vla import SpoofVLA
        vla = SpoofVLA(env.client)
    else:
        from src.smol_vla import SmolVLA
        vla = SmolVLA(env)

    # Initialize trajectory at the current EE position
    x_init, _ = env.get_ee_state()
    traj.initialize(x_init)

    # Refs held between trajectory updates
    x_des = x_init.copy()
    xdot_des = np.zeros(3)
    xddot_des = np.zeros(3)

    prev_goal = x_init.copy()
    tick = 0
    t_start = time.time()

    mode = "GUI sliders (spoof)" if args.spoof else "SmolVLA policy"
    print(f"OSC controller running with {mode}.")
    print("Close the GUI window to stop.\n")

    try:
        while True:
            # --- 20 Hz: VLA layer (read goal sliders) ---
            if tick % TICKS_PER_VLA == 0:
                goal = vla.read_goal()
                # Only re-plan if goal actually changed
                if not np.allclose(goal, prev_goal, atol=1e-3):
                    traj.set_new_goal(goal)
                    prev_goal = goal.copy()

            # --- 200 Hz: Trajectory generator ---
            if tick % TICKS_PER_TRAJ == 0:
                x_des, xdot_des, xddot_des = traj.update(DT_TRAJ)

            # --- 1000 Hz: OSC + physics ---
            _, err, debug = osc.step(env, x_des, xdot_des, xddot_des)

            # --- Periodic status print (every 2 seconds) ---
            tick += 1
            if tick % 2000 == 0:
                elapsed = time.time() - t_start
                sim_time = tick * DT_SIM
                rtf = sim_time / elapsed  # real-time factor
                F = debug["F_cmd"]
                tau = debug["tau"]
                tau_n = debug["tau_null"]
                print(
                    f"  t={sim_time:5.1f}s | err={err:.4f}m | "
                    f"F={np.round(F, 1)} N | "
                    f"tau={np.round(tau, 1)} Nm | "
                    f"tau_null={np.round(tau_n, 1)} Nm | "
                    f"RTF={rtf:.2f}x"
                )

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

    print("Done.")


if __name__ == "__main__":
    main()
