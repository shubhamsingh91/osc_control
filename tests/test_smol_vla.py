"""
Tests for SmolVLA adapter — verifies observation format, preprocessor
pipeline, and a single forward pass through the policy.

Usage:  python tests/test_smol_vla.py
"""

import numpy as np
from src.env import PandaEnv
from src.smol_vla import SmolVLA


def test_observation_format(vla):
    """Verify _build_observation returns correct keys, shapes, and dtypes."""
    import torch

    obs = vla._build_observation()

    # Expected keys
    expected_keys = {
        "observation.state",
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
        "task",
    }
    assert set(obs.keys()) == expected_keys, (
        f"Unexpected keys: {set(obs.keys())} != {expected_keys}"
    )

    # State: (6,) float32 torch tensor, no batch dim
    state = obs["observation.state"]
    assert isinstance(state, torch.Tensor), f"state type: {type(state)}"
    assert state.shape == (6,), f"state shape: {state.shape}"
    assert state.dtype == torch.float32, f"state dtype: {state.dtype}"

    # Images: (3, 256, 256) float32 torch tensor in [0, 1], no batch dim
    for key in ["observation.images.camera1",
                "observation.images.camera2",
                "observation.images.camera3"]:
        img = obs[key]
        assert isinstance(img, torch.Tensor), f"{key} type: {type(img)}"
        assert img.shape == (3, 256, 256), f"{key} shape: {img.shape}"
        assert img.dtype == torch.float32, f"{key} dtype: {img.dtype}"
        assert 0.0 <= img.min() and img.max() <= 1.0, (
            f"{key} range: [{img.min()}, {img.max()}]"
        )

    # Task: plain string
    assert isinstance(obs["task"], str), f"task type: {type(obs['task'])}"

    print("[OK] _build_observation: keys, shapes, dtypes all correct")


def test_preprocessor(vla):
    """Verify the preprocessor converts raw obs into batched tensors."""
    import torch

    obs = vla._build_observation()
    batch = vla.preprocess(obs)

    # After preprocessing: images should be tensors with batch dim
    for key in ["observation.images.camera1",
                "observation.images.camera2",
                "observation.images.camera3"]:
        assert key in batch, f"Missing {key} after preprocessing"
        img = batch[key]
        assert isinstance(img, torch.Tensor), f"{key} not a tensor: {type(img)}"
        assert img.ndim == 4, f"{key} should be 4D (b,c,h,w), got {img.shape}"
        assert img.shape[0] == 1, f"{key} batch dim should be 1, got {img.shape[0]}"
        print(f"  {key}: {img.shape} {img.dtype} on {img.device}")

    # State should be a tensor with batch dim
    state = batch["observation.state"]
    assert isinstance(state, torch.Tensor), f"state not a tensor: {type(state)}"
    assert state.ndim == 2, f"state should be 2D (b, d), got {state.shape}"
    print(f"  observation.state: {state.shape} {state.dtype}")

    # Task should be tokenized
    assert "observation.language.tokens" in batch, "Missing tokenized task"
    tokens = batch["observation.language.tokens"]
    assert isinstance(tokens, torch.Tensor), f"tokens not a tensor: {type(tokens)}"
    print(f"  observation.language.tokens: {tokens.shape} {tokens.dtype}")

    print("[OK] Preprocessor produces correct batched tensors")


def test_forward_pass(vla):
    """Run a single forward pass and verify action output."""
    action = vla._predict_action()

    assert isinstance(action, np.ndarray), f"action type: {type(action)}"
    assert action.ndim == 1, f"action should be 1D, got shape {action.shape}"
    assert action.size >= 3, f"action should have >= 3 dims, got {action.size}"
    print(f"  action shape: {action.shape}")
    print(f"  action values: {np.round(action[:6], 4)}")

    print("[OK] Forward pass produces valid action")


def test_read_goal(vla):
    """Verify read_goal returns a clipped 3D position."""
    goal = vla.read_goal()

    assert isinstance(goal, np.ndarray), f"goal type: {type(goal)}"
    assert goal.shape == (3,), f"goal shape: {goal.shape}"

    # Should be within workspace bounds
    ws = vla.workspace
    for i, axis in enumerate(["x", "y", "z"]):
        assert ws[i, 0] <= goal[i] <= ws[i, 1], (
            f"goal[{axis}]={goal[i]:.4f} outside workspace "
            f"[{ws[i,0]}, {ws[i,1]}]"
        )

    print(f"  goal: {np.round(goal, 4)}")
    print("[OK] read_goal returns valid workspace-clipped position")


def main():
    env = PandaEnv(dt=1.0 / 1000.0, gui=False)

    print("Loading SmolVLA policy (this may take a moment)...")
    vla = SmolVLA(env)
    print(f"Policy loaded on device: {vla.device}\n")

    print("--- Test 1: Observation format ---")
    test_observation_format(vla)

    print("\n--- Test 2: Preprocessor pipeline ---")
    test_preprocessor(vla)

    print("\n--- Test 3: Forward pass ---")
    test_forward_pass(vla)

    print("\n--- Test 4: read_goal ---")
    test_read_goal(vla)

    env.close()
    print("\nAll SmolVLA tests passed!")


if __name__ == "__main__":
    main()
