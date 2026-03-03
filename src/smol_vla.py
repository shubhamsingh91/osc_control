"""
SmolVLA-backed goal source for the OSC stack.

This adapter runs SmolVLA inference and converts the predicted 6D action
into a position delta for the OSC end-effector target.

Assumptions for this project:
  - State input is EE position + EE linear velocity (6D).
  - Three fixed simulation camera views are used as image inputs.
  - Action[:3] is interpreted as Cartesian delta command.
"""

import numpy as np
import pybullet as p


class SmolVLA:
    def __init__(
        self,
        env,
        model_id="lerobot/smolvla_base",
        task="Move the robot end-effector to the requested target.",
        device=None,
        action_scale=0.03,
        workspace=None,
    ):
        """
        Args:
            env: PandaEnv instance.
            model_id: Hugging Face model id/path for SmolVLA checkpoint.
            task: Natural language instruction fed to the policy.
            device: Torch device string ("cuda", "cpu"), auto if None.
            action_scale: Meters per VLA action unit for Cartesian delta.
            workspace: (3,2) XYZ limits [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
        """
        self.env = env
        self.client = env.client
        self.model_id = model_id
        self.task = task
        self.action_scale = float(action_scale)
        self.workspace = np.array(
            workspace
            if workspace is not None
            else [[-0.55, 0.80], [-0.50, 0.50], [0.05, 0.85]],
            dtype=float,
        )

        self._load_policy(device=device)

        x0, _ = self.env.get_ee_state()
        self._last_goal = x0.copy()
        self._marker_id = self._create_goal_marker(x0)

    def _load_policy(self, device=None):
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError:
            try:
                import torch
                from lerobot.policies.factory import make_pre_post_processors
                from lerobot.common.policies.smolvla.modeling_smolvla import (
                    SmolVLAPolicy,
                )
            except ImportError as exc:
                raise ImportError(
                    "SmolVLA dependencies are missing. Install with "
                    "`pip install \"lerobot[smolvla]\"`."
                ) from exc

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.policy = SmolVLAPolicy.from_pretrained(self.model_id).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def _create_goal_marker(self, pos):
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 0.7],
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=pos.tolist(),
            physicsClientId=self.client,
        )

    def _to_chw_tensor(self, rgb):
        arr = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        return self._torch.from_numpy(arr)

    def _build_observation(self):
        x, xdot = self.env.get_ee_state()
        state = self._torch.from_numpy(
            np.concatenate([x, xdot]).astype(np.float32)
        )

        img1 = self.env.get_camera_rgb("front", width=256, height=256)
        img2 = self.env.get_camera_rgb("side", width=256, height=256)
        img3 = self.env.get_camera_rgb("top", width=256, height=256)

        return {
            "observation.state": state,
            "observation.images.camera1": self._to_chw_tensor(img1),
            "observation.images.camera2": self._to_chw_tensor(img2),
            "observation.images.camera3": self._to_chw_tensor(img3),
            "task": self.task,
        }

    @staticmethod
    def _to_numpy_action(action):
        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]
            else:
                action = next(iter(action.values()))

        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()

        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _predict_action(self):
        obs = self._build_observation()
        with self._torch.inference_mode():
            processed = self.preprocess(obs)
            action = self.policy.select_action(processed)
            action = self.postprocess(action)

        action_np = self._to_numpy_action(action)
        if action_np.size < 3:
            raise RuntimeError(f"SmolVLA action has unexpected shape: {action_np.shape}")
        return action_np

    def read_goal(self):
        """
        Returns:
            goal: (3,) target EE position [m]
        """
        x, _ = self.env.get_ee_state()
        action = self._predict_action()

        # Use the first 3 action dims as Cartesian displacement command.
        delta = self.action_scale * np.tanh(action[:3])
        goal = np.clip(x + delta, self.workspace[:, 0], self.workspace[:, 1])

        if not np.allclose(goal, self._last_goal, atol=1e-4):
            p.resetBasePositionAndOrientation(
                self._marker_id,
                goal.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.client,
            )
            self._last_goal = goal.copy()

        return goal
