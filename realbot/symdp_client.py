from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(PROJECT_ROOT))

try:
    import typing_extensions as _typing_extensions

    if not hasattr(_typing_extensions, "override"):
        _typing_extensions.override = lambda func: func
except Exception:
    pass


@dataclass
class FR3Config:
    robot_id: str = "fr3"
    robot_ip: str = "172.16.0.2"
    load_gripper: bool = True
    relative_dynamics_factor: float = 0.05
    buffer_size: int = 10
    home: bool = True
    initial_end_pose: tuple[float, float, float, float, float, float, float] = (
        0.4707543519985666,
        -0.02776023399946076,
        0.29950666236991815,
        1.0,
        0.0,
        0.0,
        0.0,
    )

    scene_camera_id: Optional[str] = None
    wrist_camera_id: Optional[str] = "112322074840"
    fps: int = 15
    width: int = 640
    height: int = 480
    camera_buffer: int = 5
    img_update_rate: int = 15
    image_size: int = 84
    color_order: str = "rgb"

    action_mode: str = "POSITION_DELTA"
    asynchronous: bool = False


@dataclass
class TaskConfig:
    server_host: str = "10.184.17.132"
    server_port: int = 5001
    run_seconds: float = 600.0
    control_hz: int = 5
    execute_actions: bool = True
    action_horizon: int = 8
    merge_count: int = 4
    gripper_close_threshold: float = 0.9


ACTION_INPUT_MODE = "relative_traj"


def frame_field(frame: Any, *names: str) -> Any:
    for name in names:
        if hasattr(frame, name):
            value = getattr(frame, name)
            if value is not None:
                return value
        if isinstance(frame, dict) and frame.get(name) is not None:
            return frame[name]
    return None


def image_to_rgb(image: np.ndarray, color_order: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC/CHW image, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if color_order == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_order != "rgb":
        raise ValueError(f"Unsupported color order: {color_order}")
    return image.astype(np.uint8)


def resize_rgb_hwc(image: np.ndarray, size: int, color_order: str) -> np.ndarray:
    rgb = image_to_rgb(image, color_order)
    if rgb.shape[:2] != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def parse_eef_state(state: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(getattr(state, "end_effector_position"), dtype=np.float32).reshape(-1)
    if pose.size < 7:
        raise ValueError(f"end_effector_position must have 7 values, got {pose.shape}")
    pos = pose[:3].astype(np.float32)
    qw, qx, qy, qz = pose[3:7]
    quat_xyzw = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        raise ValueError("EEF quaternion norm is zero.")
    return pos, quat_xyzw / norm


def gripper_qpos_from_width(width: Any) -> np.ndarray:
    width = float(np.asarray(width, dtype=np.float32).reshape(-1)[0])
    return np.asarray([0.5 * width, 0.5 * width], dtype=np.float32)


def pose_to_matrix(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    matrix[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return matrix


def relative_traj_to_matrix(relative_traj: np.ndarray) -> np.ndarray:
    relative_traj = np.asarray(relative_traj, dtype=np.float64).reshape(-1)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_rotvec(relative_traj[3:6]).as_matrix()
    matrix[:3, 3] = relative_traj[:3]
    return matrix


def matrix_to_relative_traj(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    pos = matrix[:3, 3].astype(np.float32)
    rotvec = R.from_matrix(matrix[:3, :3]).as_rotvec().astype(np.float32)
    return np.concatenate([pos, rotvec], axis=0)


class SYMDPPolicyClient:
    def __init__(self, cfg: TaskConfig) -> None:
        from utils import websocket_client_policy

        self.cfg = cfg
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(cfg.server_host, cfg.server_port)
        self.server_metadata = self.policy_client.get_server_metadata()

    def inference(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        request = dict(observation)
        result = self.policy_client.infer(request)
        return (
            np.asarray(result["actions"], dtype=np.float32),
            np.asarray(result["predicted_trajs"], dtype=np.float32),
        )


class SYMDPRealBotClient:
    def __init__(self, robot_cfg: FR3Config, task_cfg: TaskConfig) -> None:
        self.robot_cfg = robot_cfg
        self.task_cfg = task_cfg
        self.policy_client = SYMDPPolicyClient(task_cfg)
        self.server_metadata = self.policy_client.server_metadata
        self._validate_server_metadata()

        self.scene_camera = None
        self.wrist_camera = None
        self.robot = None
        self.scene_color = None
        self.wrist_color = None
        self.current_gripper_state = 0
        self._sensor_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        self._load_action_mapping()
        self.setup_hardware()
        print(f"SYMDP server metadata: {self.server_metadata}")
        print(f"SYMDP action input mode: {ACTION_INPUT_MODE}")

    def _validate_server_metadata(self) -> None:
        server_mode = self.server_metadata.get("action_input")
        if server_mode is not None and server_mode != ACTION_INPUT_MODE:
            raise ValueError(
                f"SYMDP realbot client only supports {ACTION_INPUT_MODE} actions, "
                f"but server action_input is {server_mode!r}."
            )
        action_shape = tuple(self.server_metadata.get("action_shape", ()))
        if action_shape and action_shape != (7,):
            raise ValueError(
                f"SYMDP realbot client only supports 7D {ACTION_INPUT_MODE} actions, "
                f"but server action_shape is {action_shape}."
            )

    def _load_action_mapping(self) -> None:
        try:
            from algo.utils.action_mapping import (  # type: ignore
                absolute_position_action_mapping,
                delta_absolute_position_action_mapping,
            )
        except Exception as exc:
            raise ImportError("Could not import algo.utils.action_mapping in the robot environment.") from exc

        self.action_mapping = {
            "POSITION_DELTA": delta_absolute_position_action_mapping.__get__(self),
            "POSITION_ABSOLUTE": absolute_position_action_mapping.__get__(self),
        }
        if self.robot_cfg.action_mode not in self.action_mapping:
            raise ValueError(f"{ACTION_INPUT_MODE} actions require a cartesian position action mode.")

    def setup_hardware(self) -> None:
        from roby.hardware.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
        from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig

        robot_conf = FR3RobotConfig(
            id=self.robot_cfg.robot_id,
            robot_ip=self.robot_cfg.robot_ip,
            load_gripper=self.robot_cfg.load_gripper,
            relative_dynamics_factor=self.robot_cfg.relative_dynamics_factor,
            buffer_size=self.robot_cfg.buffer_size,
            initial_joint=None,
            initial_end_pose=list(self.robot_cfg.initial_end_pose),
        )
        self.robot = FR3Robot(robot_conf)
        self.robot.connect()
        state = self.robot.read_state()
        self.robot._start_read_thread()
        try:
            width = float(np.asarray(getattr(state, "gripper_width"), dtype=np.float32).reshape(-1)[0])
            self.current_gripper_state = 1 if width < 0.02 else 0
        except Exception:
            self.current_gripper_state = 0

        if self.robot_cfg.home:
            self.robot.home()
            self.robot.gripper.open(0.1)
            self.current_gripper_state = 0

        def create_camera(serial: Optional[str]):
            if serial is None:
                return None
            cam_cfg = RealSenseCameraConfig(
                fps=self.robot_cfg.fps,
                width=self.robot_cfg.width,
                height=self.robot_cfg.height,
                buffer_size=self.robot_cfg.camera_buffer,
                serial_number_or_name=serial,
            )
            cam = RealSenseCamera(cam_cfg)
            cam.connect()
            cam._start_read_thread()
            return cam

        self.scene_camera = create_camera(self.robot_cfg.scene_camera_id)
        self.wrist_camera = create_camera(self.robot_cfg.wrist_camera_id)

    def update_sensors(self) -> None:
        period = 1.0 / max(1, int(self.robot_cfg.img_update_rate))
        next_t = time.time()
        while not self._stop_event.is_set():
            scene_color = None
            wrist_color = None
            if self.scene_camera and not self.scene_camera.frame_buffer.empty():
                scene_color = frame_field(self.scene_camera.frame_buffer.queue[-1], "color", "rgb", "image")
            if self.wrist_camera and not self.wrist_camera.frame_buffer.empty():
                wrist_color = frame_field(self.wrist_camera.frame_buffer.queue[-1], "color", "rgb", "image")

            with self._sensor_lock:
                if scene_color is not None:
                    self.scene_color = np.asarray(scene_color).copy()
                if wrist_color is not None:
                    self.wrist_color = np.asarray(wrist_color).copy()

            next_t += period
            time.sleep(max(next_t - time.time(), 0.001))

    def read_observation(self) -> dict[str, Any]:
        with self._sensor_lock:
            scene_color = None if self.scene_color is None else self.scene_color.copy()
            wrist_color = None if self.wrist_color is None else self.wrist_color.copy()
        if wrist_color is None:
            raise RuntimeError("Waiting for wrist camera color frames.")

        state = self.robot.read_state()
        eef_pos, eef_quat = parse_eef_state(state)
        observation: dict[str, Any] = {
            "robot0_eye_in_hand_image": resize_rgb_hwc(
                wrist_color,
                self.robot_cfg.image_size,
                self.robot_cfg.color_order,
            ),
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": eef_quat,
            "robot0_gripper_qpos": gripper_qpos_from_width(getattr(state, "gripper_width")),
            "gripper_width": float(np.asarray(getattr(state, "gripper_width")).reshape(-1)[0]),
        }
        if scene_color is not None:
            observation["agentview_image"] = resize_rgb_hwc(
                scene_color,
                self.robot_cfg.image_size,
                self.robot_cfg.color_order,
            )
        return observation

    def _gripper_command_to_target(self, command: float) -> float:
        command = float(command)
        if command >= self.task_cfg.gripper_close_threshold:
            return 1.0
        if command <= self.task_cfg.gripper_close_threshold:
            return 0.0
        return float(self.current_gripper_state)

    def _relative_action_to_absolute_pose(
        self,
        action: np.ndarray,
        gripper_pose: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, R]:
        gripper_pos, gripper_quat = gripper_pose
        abs_pose = pose_to_matrix(gripper_pos, gripper_quat) @ relative_traj_to_matrix(action)
        return abs_pose[:3, 3].astype(np.float32), R.from_matrix(abs_pose[:3, :3])

    def convert_action_for_robot(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7:
            raise ValueError(f"SYMDP {ACTION_INPUT_MODE} action expects 7 values, got {action.size}")

        gripper_target = self._gripper_command_to_target(action[-1])
        state = self.robot.read_state()
        current_pos, current_quat = parse_eef_state(state)
        target_pos, target_rot = self._relative_action_to_absolute_pose(action, (current_pos, current_quat))

        if self.robot_cfg.action_mode == "POSITION_DELTA":
            delta_pos = target_pos - current_pos
            delta_quat = (R.from_quat(current_quat).inv() * target_rot).as_quat()
            return np.concatenate([delta_pos, delta_quat, [gripper_target]]).astype(np.float32)

        if self.robot_cfg.action_mode == "POSITION_ABSOLUTE":
            euler_deg = target_rot.as_euler("xyz", degrees=True)
            return np.concatenate([target_pos, euler_deg, [gripper_target]]).astype(np.float32)

        raise ValueError(f"{ACTION_INPUT_MODE} actions require a cartesian position action mode.")

    def merge_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.size == 0:
            raise ValueError("No actions returned by SYMDP policy.")

        merge_count = max(1, int(self.task_cfg.merge_count))
        if merge_count == 1 or len(actions) < merge_count:
            return actions

        merged = []
        for start in range(0, len(actions), merge_count):
            chunk = actions[start : start + merge_count]
            relative_pose = np.eye(4, dtype=np.float64)
            for step in chunk:
                relative_pose = relative_pose @ relative_traj_to_matrix(step)
            pose_6d = matrix_to_relative_traj(relative_pose)

            gripper_commands = chunk[:, 6]
            active = gripper_commands[np.abs(gripper_commands) >= self.task_cfg.gripper_close_threshold]
            gripper = np.asarray([active[-1] if active.size else 0.0], dtype=np.float32)
            merged.append(np.concatenate([pose_6d, gripper], axis=0))
        return np.asarray(merged, dtype=np.float32)

    def move(
        self,
        action: np.ndarray,
        execute: bool = True,
    ) -> None:
        robot_action = self.convert_action_for_robot(action)
        if self.robot_cfg.action_mode == "POSITION_DELTA":
            delta_rotvec = R.from_quat(robot_action[3:7]).as_rotvec()
            print("SYMDP robot delta action:", np.concatenate([robot_action[:3], delta_rotvec, robot_action[-1:]]))
        else:
            print("SYMDP robot action:", robot_action)

        gripper_state_before_mapping = self.current_gripper_state
        mapped = self.action_mapping[self.robot_cfg.action_mode](robot_action)
        if not execute:
            self.current_gripper_state = gripper_state_before_mapping
        if getattr(mapped, "cartesian_positions", None) is not None:
            print(f"FR3 mapped cartesian ({mapped.action_mode}): {np.asarray(mapped.cartesian_positions)}")
        elif getattr(mapped, "joint_positions", None) is not None:
            print(f"FR3 mapped joints ({mapped.action_mode}): {np.asarray(mapped.joint_positions)}")
        if not execute:
            print("DRY RUN: not sending action to robot.")
            return
        self.robot.send_action(mapped, asynchronous=self.robot_cfg.asynchronous)

    def control_loop(self) -> None:
        period = 1.0 / max(1, int(self.task_cfg.control_hz))
        next_t = time.time()
        while not self._stop_event.is_set():
            try:
                obs = self.read_observation()
                actions, _ = self.policy_client.inference(obs)
                actions = self.merge_actions(actions[: self.task_cfg.action_horizon])
                print(f"SYMDP merged actions: {len(actions)} from horizon {self.task_cfg.action_horizon}")
                print(f"SYMDP merged action[0]: {actions[0]}")
                print(f"SYMDP merged action[1]: {actions[1]}")
                for action in actions:
                    self.move(action, execute=self.task_cfg.execute_actions)
            except RuntimeError as exc:
                print(exc)
            except Exception as exc:
                logging.exception("SYMDP control loop failed: %s", exc)

            next_t += period
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.time()

    def run(self) -> None:
        self._threads = [
            threading.Thread(target=self.update_sensors, name="symdp_update_sensors", daemon=True),
            threading.Thread(target=self.control_loop, name="symdp_control_loop", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        start = time.time()
        try:
            while True:
                if self.task_cfg.run_seconds and time.time() - start > self.task_cfg.run_seconds:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt.")
        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for cam in (self.scene_camera, self.wrist_camera):
            try:
                if cam is not None:
                    cam.disconnect()
            except Exception:
                pass
        try:
            if self.robot is not None:
                self.robot.disconnect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealBot client for SYMDP websocket policy inference.")
    parser.add_argument("--server-host", default=TaskConfig.server_host)
    parser.add_argument("--server-port", type=int, default=TaskConfig.server_port)
    parser.add_argument("--run-seconds", type=float, default=TaskConfig.run_seconds)
    parser.add_argument("--control-hz", type=int, default=TaskConfig.control_hz)
    parser.add_argument("--dry-run", action="store_true", help="Run inference but do not send actions to the robot.")
    parser.add_argument("--action-horizon", type=int, default=TaskConfig.action_horizon)
    parser.add_argument("--merge-count", type=int, default=TaskConfig.merge_count)
    parser.add_argument("--gripper-close-threshold", type=float, default=TaskConfig.gripper_close_threshold)

    parser.add_argument("--robot-ip", default=FR3Config.robot_ip)
    parser.add_argument("--scene-camera-id", default=FR3Config.scene_camera_id)
    parser.add_argument("--wrist-camera-id", default=FR3Config.wrist_camera_id)
    parser.add_argument("--image-size", type=int, default=FR3Config.image_size)
    parser.add_argument("--color-order", choices=("rgb", "bgr"), default=FR3Config.color_order)
    parser.add_argument(
        "--action-mode",
        choices=("POSITION_DELTA", "POSITION_ABSOLUTE"),
        default=FR3Config.action_mode,
    )
    parser.add_argument("--no-home", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    robot_cfg = FR3Config(
        robot_ip=args.robot_ip,
        scene_camera_id=args.scene_camera_id,
        wrist_camera_id=args.wrist_camera_id,
        image_size=args.image_size,
        color_order=args.color_order,
        action_mode=args.action_mode,
        home=not args.no_home,
    )
    task_cfg = TaskConfig(
        server_host=args.server_host,
        server_port=args.server_port,
        run_seconds=args.run_seconds,
        control_hz=args.control_hz,
        execute_actions=not args.dry_run,
        action_horizon=args.action_horizon,
        merge_count=args.merge_count,
        gripper_close_threshold=args.gripper_close_threshold,
    )
    SYMDPRealBotClient(robot_cfg, task_cfg).run()


if __name__ == "__main__":
    main()
