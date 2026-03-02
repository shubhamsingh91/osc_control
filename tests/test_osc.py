"""
Test for osc.py — verify the controller drives the EE to a target position.

Runs a 3-second sim with a fixed target, checks convergence.
Usage:  conda activate pybullet_env && python test_osc.py
"""

import numpy as np
from src.env import PandaEnv
from src.osc import OSC


def main():
    env = PandaEnv(dt=1.0 / 1000.0, gui=False)
    osc = OSC()

    # Target: 10 cm forward and 10 cm up from home EE position
    pos_home, _ = env.get_ee_state()
    x_des = pos_home + np.array([0.1, 0.0, 0.1])
    xdot_des = np.zeros(3)
    xddot_des = np.zeros(3)

    print(f"Home EE pos:   {np.round(pos_home, 4)}")
    print(f"Target EE pos: {np.round(x_des, 4)}")

    # Run for 3 seconds (3000 ticks at 1 kHz)
    errors = []
    for i in range(3000):
        x, err, _ = osc.step(env, x_des, xdot_des, xddot_des)
        errors.append(err)

        if i % 500 == 0:
            print(f"  t={i/1000:.1f}s  pos={np.round(x, 4)}  err={err:.4f} m")

    final_err = errors[-1]
    print(f"\nFinal error: {final_err:.6f} m")
    assert final_err < 0.005, f"OSC did not converge: final err = {final_err:.4f} m"
    print("[OK] OSC converged to target (< 5 mm error)")

    # Check that error decreased monotonically (roughly — allow minor overshoot)
    peak_err = max(errors[:100])   # initial transient
    final_err = np.mean(errors[-100:])  # steady state
    print(f"Peak error (first 100ms): {peak_err:.4f} m")
    print(f"Mean error (last 100ms):  {final_err:.6f} m")
    assert final_err < peak_err, "Controller didn't improve over time!"
    print("[OK] Error decreased from transient to steady state")

    env.close()
    print("\nAll OSC tests passed!")


if __name__ == "__main__":
    main()
