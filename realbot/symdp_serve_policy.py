from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = "/data1/user/rensihua/sym_in_dp/data/outputs/2026.05.21/13.58.30_diff_c_real_cake_box/checkpoints/epoch=0590-train_loss=0.006.ckpt"
ACTION_INPUT_MODE = "relative_traj"
ACTION_FORMULA = "T_abs = T_gripper @ T_relative_traj"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(PROJECT_ROOT))

from websocket import base_policy as _base_policy  # noqa: E402


def shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in shape)


def require_cv2():
    import cv2

    return cv2


def as_float_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        raise KeyError(f"Missing observation field: {name}")
    return np.asarray(value, dtype=np.float32).reshape(shape)


def normalize_quat_xyzw(quat: Any) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat_arr)
    if norm < 1e-8:
        raise ValueError("Quaternion norm is zero.")
    return quat_arr / norm


def pos_from_observation(obs: dict[str, Any]) -> np.ndarray:
    if "robot0_eef_pos" in obs:
        return as_float_array(obs["robot0_eef_pos"], "robot0_eef_pos", (3,))
    if "eef_pos" in obs:
        return as_float_array(obs["eef_pos"], "eef_pos", (3,))
    if "eef_pose" in obs:
        pose = np.asarray(obs["eef_pose"], dtype=np.float32).reshape(-1)
        if pose.size < 3:
            raise ValueError("eef_pose must contain at least x y z.")
        return pose[:3].astype(np.float32)
    if "gripper_state" in obs:
        pose = np.asarray(obs["gripper_state"], dtype=np.float32).reshape(-1)
        if pose.size < 3:
            raise ValueError("gripper_state must contain at least x y z.")
        return pose[:3].astype(np.float32)
    raise KeyError("Missing robot0_eef_pos/eef_pos/eef_pose/gripper_state.")


def quat_from_observation(obs: dict[str, Any]) -> np.ndarray:
    if "robot0_eef_quat" in obs:
        return normalize_quat_xyzw(obs["robot0_eef_quat"])
    if "eef_quat" in obs:
        return normalize_quat_xyzw(obs["eef_quat"])
    if "eef_pose" in obs:
        pose = np.asarray(obs["eef_pose"], dtype=np.float32).reshape(-1)
        if pose.size < 7:
            raise ValueError("eef_pose must contain x y z qw qx qy qz.")
        qw, qx, qy, qz = pose[3:7]
        return normalize_quat_xyzw([qx, qy, qz, qw])
    raise KeyError("Missing robot0_eef_quat/eef_quat/eef_pose.")


def gripper_qpos_from_observation(obs: dict[str, Any], mode: str) -> np.ndarray:
    if "robot0_gripper_qpos" in obs:
        return as_float_array(obs["robot0_gripper_qpos"], "robot0_gripper_qpos", (2,))
    if "gripper_qpos" in obs:
        return as_float_array(obs["gripper_qpos"], "gripper_qpos", (2,))

    width = float(np.asarray(obs.get("gripper_width", 0.0), dtype=np.float32).reshape(-1)[0])
    if mode == "half_width":
        return np.asarray([0.5 * width, 0.5 * width], dtype=np.float32)
    if mode == "duplicate_width":
        return np.asarray([width, width], dtype=np.float32)
    if mode == "width_zero":
        return np.asarray([width, 0.0], dtype=np.float32)
    raise ValueError(f"Unsupported gripper qpos mode: {mode}")


def image_from_observation(obs: dict[str, Any], key: str, expected_shape: tuple[int, int, int]) -> np.ndarray:
    value = obs.get(key)
    if value is None and key == "robot0_eye_in_hand_image":
        value = obs.get("wrist_image", obs.get("eye_image"))
    if value is None and key == "agentview_image":
        value = obs.get("left_image", obs.get("scene_image"))
    if value is None:
        raise KeyError(f"Missing image observation field: {key}")

    image = np.asarray(value)
    expected_c, expected_h, expected_w = expected_shape
    if image.ndim == 2:
        image = require_cv2().cvtColor(image, require_cv2().COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise ValueError(f"{key} must be an image with 3 dims, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[:2] != (expected_h, expected_w):
        image = require_cv2().resize(image, (expected_w, expected_h), interpolation=require_cv2().INTER_AREA)
    if image.shape[-1] < expected_c:
        pad = np.zeros((*image.shape[:2], expected_c - image.shape[-1]), dtype=image.dtype)
        image = np.concatenate([image, pad], axis=-1)
    elif image.shape[-1] > expected_c:
        image = image[..., :expected_c]

    image = image.astype(np.float32)
    if image.size and np.nanmax(image) > 1.5:
        image /= 255.0
    return np.moveaxis(image, -1, 0).astype(np.float32)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_policy_from_checkpoint(checkpoint: Path, device: Any):
    import dill
    import hydra
    import torch

    checkpoint = checkpoint.expanduser()
    try:
        payload = torch.load(str(checkpoint), pickle_module=dill, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint), pickle_module=dill, map_location="cpu")

    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(checkpoint.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return cfg, policy


class SYMDPWebsocketPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        gripper_qpos_mode: str,
    ) -> None:
        import torch
        from omegaconf import OmegaConf
        from sym_in_dp.model.common.rotation_transformer import RotationTransformer

        self.torch = torch
        self.device = torch.device(device)
        self.cfg, self.policy = load_policy_from_checkpoint(checkpoint, self.device)
        self.shape_meta = OmegaConf.to_container(self.cfg.shape_meta, resolve=True)
        self.obs_shape_meta = self.shape_meta["obs"]
        self.action_shape = shape_tuple(self.shape_meta["action"]["shape"])
        if self.action_shape != (7,):
            raise ValueError(
                f"SYMDP websocket policy only supports 7D {ACTION_INPUT_MODE} actions, "
                f"got action shape {self.action_shape}."
            )
        self.n_obs_steps = int(getattr(self.policy, "n_obs_steps", self.cfg.n_obs_steps))
        self.obs_history: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)
        self.gripper_qpos_mode = gripper_qpos_mode

        self.rgb_keys = [
            key for key, attr in self.obs_shape_meta.items()
            if attr.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            key for key, attr in self.obs_shape_meta.items()
            if attr.get("type", "low_dim") == "low_dim"
        ]

    def reset(self) -> None:
        self.obs_history.clear()

    def observation_to_entry(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        entry: dict[str, np.ndarray] = {}
        for key, attr in self.obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            shape = shape_tuple(attr["shape"])
            if obs_type == "rgb":
                entry[key] = image_from_observation(obs, key, shape)
            elif key.endswith("eef_pos"):
                entry[key] = pos_from_observation(obs)
            elif key.endswith("eef_quat"):
                entry[key] = quat_from_observation(obs)
            elif key.endswith("gripper_qpos"):
                entry[key] = gripper_qpos_from_observation(obs, self.gripper_qpos_mode)
            elif key in obs:
                entry[key] = as_float_array(obs[key], key, shape)
            else:
                raise KeyError(f"Unsupported or missing observation field: {key}")
        return entry

    def history_to_torch(self) -> dict[str, Any]:
        result = {}
        for key in self.obs_history[0].keys():
            arr = np.stack([entry[key] for entry in self.obs_history], axis=0)[None]
            result[key] = self.torch.from_numpy(arr).to(self.device, dtype=self.torch.float32)
        return result

    def actions_to_7d(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        action_dim = actions.shape[-1]
        if action_dim != 7:
            raise ValueError(f"SYMDP {ACTION_INPUT_MODE} action dimension must be 7, got {action_dim}.")
        return actions.astype(np.float32)

    @property
    def action_input_mode(self) -> str:
        return ACTION_INPUT_MODE

    @property
    def action_formula(self) -> str:
        return ACTION_FORMULA

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        entry = self.observation_to_entry(obs)
        if len(self.obs_history) == 0:
            for _ in range(self.n_obs_steps):
                self.obs_history.append(entry)
        else:
            self.obs_history.append(entry)

        obs_dict = self.history_to_torch()
        with self.torch.inference_mode():
            result = self.policy.predict_action(obs_dict)
            action_raw = tensor_to_numpy(result["action"])[0]
            pred_raw = tensor_to_numpy(result.get("action_pred", result["action"]))[0]

        action_7d = self.actions_to_7d(action_raw)
        pred_7d = self.actions_to_7d(pred_raw)
        response = {
            "actions": action_7d.tolist(),
            "predicted_trajs": pred_7d.tolist(),
            "n_obs_steps": self.n_obs_steps,
            "action_input": self.action_input_mode,
            "action_formula": self.action_formula,
        }
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the real-robot SYMDP policy over websocket.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gripper-qpos-mode", choices=("half_width", "duplicate_width", "width_zero"), default="half_width")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    from websocket import websocket_policy_server

    policy = SYMDPWebsocketPolicy(
        checkpoint=args.checkpoint,
        device=args.device,
        gripper_qpos_mode=args.gripper_qpos_mode,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "policy": "symdp",
            "checkpoint": str(args.checkpoint.expanduser()),
            "n_obs_steps": policy.n_obs_steps,
            "observation_keys": list(policy.obs_shape_meta.keys()),
            "rgb_keys": policy.rgb_keys,
            "lowdim_keys": policy.lowdim_keys,
            "action_shape": policy.action_shape,
            "action_input": policy.action_input_mode,
            "action_formula": policy.action_formula,
            "action_format": "relative_x relative_y relative_z relative_rotvec_x relative_rotvec_y relative_rotvec_z gripper",
        },
    )
    print(f"Serving SYMDP policy on ws://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
