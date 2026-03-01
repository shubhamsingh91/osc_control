"""
Smoke test for env.py — run headless (no GUI), verify all APIs return
correct shapes and the sim doesn't explode.

Usage:  uv run python test_env.py
"""

import numpy as np
from env import PandaEnv


def main():
    env = PandaEnv(dt=1.0 / 1000.0, gui=False)

    # --- 1. Joint states ---
    q, qdot = env.get_joint_states()
    assert q.shape == (7,), f"q shape: {q.shape}"
    assert qdot.shape == (7,), f"qdot shape: {qdot.shape}"
    print(f"[OK] get_joint_states()")
    print(f"     q    = {np.round(q, 3)}")
    print(f"     qdot = {np.round(qdot, 3)}")

    # --- 2. EE state ---
    pos, vel = env.get_ee_state()
    assert pos.shape == (3,), f"pos shape: {pos.shape}"
    assert vel.shape == (3,), f"vel shape: {vel.shape}"
    print(f"\n[OK] get_ee_state()")
    print(f"     pos = {np.round(pos, 4)}")
    print(f"     vel = {np.round(vel, 4)}")

    # --- 3. Dynamics ---
    M, h = env.get_dynamics(q, qdot)
    assert M.shape == (7, 7), f"M shape: {M.shape}"
    assert h.shape == (7,), f"h shape: {h.shape}"
    # M should be symmetric positive-definite
    sym_err = np.max(np.abs(M - M.T))
    eigvals = np.linalg.eigvalsh(M)
    print(f"\n[OK] get_dynamics()")
    print(f"     M symmetry error: {sym_err:.2e}")
    print(f"     M eigenvalues:    {np.round(eigvals, 4)}")
    print(f"     h (bias forces):  {np.round(h, 4)}")
    assert sym_err < 1e-10, "M is not symmetric!"
    assert np.all(eigvals > 0), "M is not positive-definite!"

    # --- 4. Jacobian ---
    J = env.get_jacobian(q)
    assert J.shape == (3, 7), f"J shape: {J.shape}"
    print(f"\n[OK] get_jacobian()")
    print(f"     J (3x7):\n{np.round(J, 4)}")

    # --- 5. Jdot*qdot ---
    # Use a non-zero qdot to get a meaningful Jdot*qdot
    qdot_test = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.1])
    Jdot_qdot = env.get_jdot_qdot(q, qdot_test)
    assert Jdot_qdot.shape == (3,), f"Jdot_qdot shape: {Jdot_qdot.shape}"

    # Verify against finite difference:  dJ/dt @ qdot ≈ (J(q+qdot*dt) - J(q)) / dt @ qdot
    dt_fd = 1e-6
    q_perturbed = q + qdot_test * dt_fd
    J_now = env.get_jacobian(q)
    J_next = env.get_jacobian(q_perturbed)
    Jdot_fd = (J_next - J_now) / dt_fd
    Jdot_qdot_fd = Jdot_fd @ qdot_test

    fd_err = np.linalg.norm(Jdot_qdot - Jdot_qdot_fd)
    print(f"\n[OK] get_jdot_qdot()")
    print(f"     Jdot*qdot (pin):       {np.round(Jdot_qdot, 6)}")
    print(f"     Jdot*qdot (finite-diff):{np.round(Jdot_qdot_fd, 6)}")
    print(f"     error norm:             {fd_err:.2e}")
    assert fd_err < 1e-3, f"Jdot*qdot doesn't match finite-diff: err={fd_err:.4e}"

    # --- 6. Apply gravity-comp torques and step for 1 second ---
    print(f"\n--- Stepping sim for 1s with gravity compensation ---")
    pos_before = pos.copy()
    for _ in range(1000):
        q, qdot = env.get_joint_states()
        _, h = env.get_dynamics(q, qdot)
        env.apply_torques(h)  # compensate gravity + Coriolis
        env.step()

    pos_after, _ = env.get_ee_state()
    drift = np.linalg.norm(pos_after - pos_before)
    print(f"     EE before: {np.round(pos_before, 4)}")
    print(f"     EE after:  {np.round(pos_after, 4)}")
    print(f"     EE drift:  {drift:.6f} m")
    assert drift < 0.01, f"EE drifted too much under gravity comp: {drift:.4f} m"
    print(f"[OK] Gravity compensation holds pose (drift < 1 cm)")

    env.close()
    print(f"\nAll tests passed!")


if __name__ == "__main__":
    main()
