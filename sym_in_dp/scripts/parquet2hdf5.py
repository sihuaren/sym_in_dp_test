import argparse
from pathlib import Path
from typing import Any, Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_hdf5_output_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists() and path.is_dir():
        return path / "sym_realbot_image_rel_traj.hdf5"
    if path.suffix != ".hdf5":
        return path.with_suffix(".hdf5")
    return path


def list_parquet_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file must be a parquet file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return sorted(path.glob(pattern))


def decode_image(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        image = value
        if image.ndim == 3:
            return image
        if image.ndim == 1 and image.dtype == np.uint8:
            value = image.tobytes()
        else:
            return None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3:
        return None
    return image


def first_present(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def image_key_candidates(camera: str, explicit_key: Optional[str]) -> list[str]:
    if explicit_key is not None:
        return [explicit_key]
    return [
        f"{camera}/color",
        camera,
        f"{camera}/rgb",
        f"{camera}/image",
        f"{camera}_color",
        f"{camera}_rgb",
        f"{camera}_image",
    ]


def resolve_image_key(
    columns: pd.Index,
    camera: str,
    explicit_key: Optional[str],
    label: str,
) -> str:
    candidates = image_key_candidates(camera, explicit_key)
    for key in candidates:
        if key in columns:
            return key
    raise KeyError(
        f"Could not find {label} image column. Tried: {candidates}"
    )


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def pose_array_to_xyzw(
    pose_val: Any,
    rotation_rep: str,
    quat_order: str,
) -> Optional[np.ndarray]:
    if pose_val is None:
        return None
    pose_arr = np.asarray(pose_val, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] < 6:
        return None

    pos = pose_arr[:3]
    if rotation_rep == "quat":
        if pose_arr.shape[0] < 7:
            return None
        quat = pose_arr[3:7]
        if quat_order == "wxyz":
            quat_xyzw = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        elif quat_order == "xyzw":
            quat_xyzw = quat
        else:
            raise ValueError(f"Unsupported quaternion order: {quat_order}")
        quat_xyzw = normalize_quat_xyzw(quat_xyzw)
    elif rotation_rep == "rotvec":
        quat_xyzw = Rotation.from_rotvec(pose_arr[3:6]).as_quat()
    else:
        raise ValueError(f"Unsupported rotation representation: {rotation_rep}")

    return np.concatenate([pos, quat_xyzw])


def extract_pose_width(row: pd.Series, pose_keys: list[str], gripper_keys: list[str]) -> Optional[tuple[np.ndarray, float]]:
    pose = pose_array_to_xyzw(
        pose_val=first_present(row, pose_keys),
        rotation_rep="quat",
        quat_order="wxyz",
    )
    if pose is None:
        return None

    width_val = first_present(row, gripper_keys)
    width = 0.0
    if width_val is not None:
        width_arr = np.asarray(width_val, dtype=np.float64).reshape(-1)
        if width_arr.size > 0:
            width = float(width_arr[0])

    return pose, width


def action_pose_keys(args: argparse.Namespace) -> list[str]:
    if args.action_pose_key is not None:
        return [args.action_pose_key]
    return [
        f"{args.robot_prefix}/action",
        "action",
        f"{args.robot_prefix}/target_pose",
        "target_pose",
        f"{args.robot_prefix}/action_pose",
        "action_pose",
        f"{args.robot_prefix}/cartesian_action",
        "cartesian_action",
    ]


def extract_absolute_action_pose(row: pd.Series, args: argparse.Namespace) -> Optional[np.ndarray]:
    pose_val = first_present(row, action_pose_keys(args))
    return pose_array_to_xyzw(
        pose_val=pose_val,
        rotation_rep=args.action_rotation_rep,
        quat_order=args.action_quat_order,
    )


def pose_to_matrix(pose_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(pose_xyzw[3:7]).as_matrix()
    matrix[:3, 3] = pose_xyzw[:3]
    return matrix


def relative_trajectory_action(
    current_pose: np.ndarray,
    absolute_target_pose: np.ndarray,
    gripper_command: float,
    pos_scale: float,
) -> np.ndarray:
    current_t = pose_to_matrix(current_pose)
    target_t = pose_to_matrix(absolute_target_pose)
    relative_t = np.linalg.inv(current_t) @ target_t
    relative_pos = relative_t[:3, 3] * pos_scale
    relative_rotvec = Rotation.from_matrix(relative_t[:3, :3]).as_rotvec()
    return np.concatenate([relative_pos, relative_rotvec, [gripper_command]]).astype(np.float32)


def rotation_distance(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray) -> float:
    rot0 = Rotation.from_quat(q0_xyzw)
    rot1 = Rotation.from_quat(q1_xyzw)
    return float((rot1 * rot0.inv()).magnitude())


def filter_static_frames(
    rows: list[dict],
    pos_scale: float,
    eps_pos: float,
    eps_rot: float,
    gripper_threshold: float,
    keep_after_gripper: int,
) -> list[dict]:
    if not rows:
        return []

    filtered = [rows[0]]
    last_kept_pose = rows[0]["pose"]
    last_kept_width = rows[0]["width"]
    force_keep_counter = 0

    for row in rows[1:]:
        curr_pose = row["pose"]
        curr_width = row["width"]

        d_pos = np.linalg.norm((curr_pose[:3] - last_kept_pose[:3]) * pos_scale)
        d_rot = rotation_distance(last_kept_pose[3:7], curr_pose[3:7])
        d_width = abs(curr_width - last_kept_width)

        is_moving_gripper = d_width > gripper_threshold
        if is_moving_gripper:
            force_keep_counter = keep_after_gripper

        should_keep = (
            d_pos > eps_pos
            or d_rot > eps_rot
            or is_moving_gripper
            or force_keep_counter > 0
        )
        if should_keep:
            filtered.append(row)
            last_kept_pose = curr_pose
            last_kept_width = curr_width
            if force_keep_counter > 0:
                force_keep_counter -= 1

    return filtered


def gripper_commands(
    filtered_rows: list[dict],
    gripper_threshold: float,
    emit_only_on_edge: bool,
    clamp_len: int,
    debounce_steps: int,
    hold_close: bool,
) -> list[float]:
    commands = []
    last_trend = 0.0
    last_trigger_idx = -100000

    for i in range(len(filtered_rows) - 1):
        width_curr = filtered_rows[i]["width"]
        width_next = filtered_rows[i + 1]["width"]
        delta_width = width_next - width_curr

        if delta_width > gripper_threshold:
            trend = -1.0
        elif delta_width < -gripper_threshold:
            trend = 1.0
        else:
            trend = 0.0

        entering_motion = trend != 0.0 and trend != last_trend
        command = float(trend) if (not emit_only_on_edge or entering_motion) else 0.0
        last_trend = trend

        if i < clamp_len:
            command = 0.0

        if abs(command) > 0.5:
            if (i - last_trigger_idx) < debounce_steps:
                command = 0.0
            else:
                last_trigger_idx = i
        commands.append(command)

    if hold_close:
        holding = False
        for i, command in enumerate(commands):
            if command == 1.0:
                holding = True
            elif command == -1.0:
                commands[i] = 0.0
                holding = False
            elif holding:
                commands[i] = 1.0

    return commands


def resize_rgb_chw(image: Optional[np.ndarray], size: int, input_color_order: str) -> np.ndarray:
    if image is None:
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        return np.moveaxis(rgb, -1, 0)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim != 3:
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        return np.moveaxis(rgb, -1, 0)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        return np.moveaxis(rgb, -1, 0)

    if input_color_order == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif input_color_order != "rgb":
        raise ValueError(f"Unsupported image color order: {input_color_order}")

    if image.dtype == np.uint8:
        rgb = image
    else:
        rgb = image.astype(np.float32)
        finite = np.isfinite(rgb)
        if finite.any() and rgb[finite].max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.moveaxis(rgb.astype(np.uint8), -1, 0)


def gripper_qpos(width: float, mode: str) -> np.ndarray:
    if mode == "width_zero":
        return np.asarray([width, 0.0], dtype=np.float32)
    if mode == "duplicate_width":
        return np.asarray([width, width], dtype=np.float32)
    if mode == "half_width":
        return np.asarray([width * 0.5, width * 0.5], dtype=np.float32)
    raise ValueError(f"Unsupported gripper qpos mode: {mode}")


def build_episode(
    parquet_path: Path,
    args: argparse.Namespace,
) -> Optional[dict[str, np.ndarray]]:
    df = pd.read_parquet(parquet_path)

    pose_keys = [f"{args.robot_prefix}/end_effector_position", "end_effector_position"]
    gripper_keys = [f"{args.robot_prefix}/gripper_width", "gripper_width"]
    raw_rows = []
    action_pose_count = 0
    for row_pos, (_, row) in enumerate(df.iterrows()):
        pose_width = extract_pose_width(row, pose_keys, gripper_keys)
        if pose_width is None:
            continue
        pose, width = pose_width
        action_pose = extract_absolute_action_pose(row, args)
        if action_pose is not None:
            action_pose_count += 1
        raw_rows.append(
            {
                "row_pos": row_pos,
                "pose": pose,
                "width": width,
                "absolute_action_pose": action_pose,
            }
        )

    filtered_rows = filter_static_frames(
        raw_rows,
        pos_scale=args.filter_pos_scale,
        eps_pos=args.eps_pos,
        eps_rot=args.eps_rot,
        gripper_threshold=args.gripper_threshold,
        keep_after_gripper=args.keep_after_gripper,
    )
    print(
        f"[{parquet_path.name}] raw={len(raw_rows)} filtered={len(filtered_rows)} "
        f"action_pose_rows={action_pose_count}"
    )
    if len(filtered_rows) < 2:
        return None

    commands = gripper_commands(
        filtered_rows,
        gripper_threshold=args.gripper_threshold,
        emit_only_on_edge=not args.emit_every_gripper_step,
        clamp_len=args.gripper_clamp_len,
        debounce_steps=args.gripper_debounce_steps,
        hold_close=not args.no_gripper_hold_close,
    )

    final_rows = filtered_rows[:-1]
    actions = []
    eef_pos = []
    eef_quat = []
    gripper = []
    eye_images = []
    agentview_images = []

    image_key = resolve_image_key(
        df.columns,
        camera=args.eye_camera,
        explicit_key=args.eye_image_key,
        label="eye-in-hand",
    )
    agentview_image_key = resolve_image_key(
        df.columns,
        camera=args.agentview_camera,
        explicit_key=args.agentview_image_key,
        label="agentview",
    )

    for curr, nxt, command in zip(final_rows, filtered_rows[1:], commands):
        row = df.iloc[curr["row_pos"]]
        target_pose = curr["absolute_action_pose"]
        if target_pose is None:
            target_pose = nxt["pose"]

        action = relative_trajectory_action(
            current_pose=curr["pose"],
            absolute_target_pose=target_pose,
            gripper_command=command,
            pos_scale=args.action_pos_scale,
        )

        image = decode_image(row.get(image_key))
        agentview_image = decode_image(row.get(agentview_image_key))

        actions.append(action)
        eef_pos.append(curr["pose"][:3].astype(np.float32))
        eef_quat.append(curr["pose"][3:7].astype(np.float32))
        gripper.append(gripper_qpos(curr["width"], args.gripper_qpos_mode))
        eye_images.append(
            resize_rgb_chw(
                image,
                size=args.image_size,
                input_color_order=args.eye_image_color_order,
            )
        )
        agentview_images.append(
            resize_rgb_chw(
                agentview_image,
                size=args.image_size,
                input_color_order=args.agentview_image_color_order,
            )
        )

    return {
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "robot0_eef_pos": np.stack(eef_pos, axis=0).astype(np.float32),
        "robot0_eef_quat": np.stack(eef_quat, axis=0).astype(np.float32),
        "robot0_gripper_qpos": np.stack(gripper, axis=0).astype(np.float32),
        "robot0_eye_in_hand_image": np.stack(eye_images, axis=0).astype(np.uint8),
        "agentview_image": np.stack(agentview_images, axis=0).astype(np.uint8),
    }


def write_demo(data_group: h5py.Group, demo_idx: int, episode: dict[str, np.ndarray], source: Path, compression: Optional[str]) -> int:
    demo = data_group.create_group(f"demo_{demo_idx}")
    demo.attrs["source_parquet"] = str(source)
    demo.attrs["num_samples"] = int(episode["actions"].shape[0])
    demo.create_dataset("actions", data=episode["actions"], compression=compression)

    obs = demo.create_group("obs")
    obs.create_dataset("robot0_eef_pos", data=episode["robot0_eef_pos"], compression=compression)
    obs.create_dataset("robot0_eef_quat", data=episode["robot0_eef_quat"], compression=compression)
    obs.create_dataset("robot0_gripper_qpos", data=episode["robot0_gripper_qpos"], compression=compression)
    obs.create_dataset("robot0_eye_in_hand_image", data=episode["robot0_eye_in_hand_image"], compression=compression)
    obs.create_dataset("agentview_image", data=episode["agentview_image"], compression=compression)
    return int(episode["actions"].shape[0])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert real-robot parquet episodes into a SymInDP image HDF5 dataset."
    )
    parser.add_argument("--input", type=Path, default='/hard_data/user_dataset/rensihua_dataset/realbot_260518/hug_cup_260513', help="Parquet file or directory. One parquet is treated as one episode.")
    parser.add_argument("--output", type=Path, default='/hard_data/user_dataset/rensihua_dataset/realbot_260518/hdf5/hug_cup_260513_new', help="Merged output HDF5 path.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-errors", action="store_true")

    parser.add_argument("--eye-camera", default="wrist_camera")
    parser.add_argument("--eye-image-key", default=None)
    parser.add_argument(
        "--eye-image-color-order",
        choices=("bgr", "rgb"),
        default="rgb",
        help="Channel order returned by decoding the wrist image before saving as RGB.",
    )
    parser.add_argument(
        "--agentview-camera",
        default="left_camera",
        help="Parquet camera prefix for agentview_image. Defaults to left_camera.",
    )
    parser.add_argument(
        "--agentview-image-key",
        default=None,
        help="Parquet column containing the agentview image. Defaults to left_camera/color fallbacks.",
    )
    parser.add_argument(
        "--agentview-image-color-order",
        choices=("bgr", "rgb"),
        default="rgb",
        help="Channel order returned by decoding the agentview image before saving as RGB.",
    )
    parser.add_argument("--image-size", type=int, default=84)

    parser.add_argument("--robot-prefix", default="fr3")
    parser.add_argument(
        "--action-pose-key",
        default=None,
        help="Parquet column containing absolute target pose. Defaults to action/target_pose fallbacks.",
    )
    parser.add_argument("--action-rotation-rep", choices=("quat", "rotvec"), default="quat")
    parser.add_argument("--action-quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--action-pos-scale", type=float, default=1.0)

    parser.add_argument("--filter-pos-scale", type=float, default=100.0)
    parser.add_argument("--eps-pos", type=float, default=1.0)
    parser.add_argument("--eps-rot", type=float, default=1e-3)
    parser.add_argument("--gripper-threshold", type=float, default=0.015)
    parser.add_argument("--keep-after-gripper", type=int, default=5)
    parser.add_argument("--emit-every-gripper-step", action="store_true")
    parser.add_argument("--gripper-clamp-len", type=int, default=8)
    parser.add_argument("--gripper-debounce-steps", type=int, default=8)
    parser.add_argument("--no-gripper-hold-close", action="store_true")
    parser.add_argument("--gripper-qpos-mode", choices=("half_width", "duplicate_width", "width_zero"), default="half_width")

    parser.add_argument("--compression", default="gzip", choices=("gzip", "none"))
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    parquet_files = list_parquet_files(args.input.expanduser(), args.recursive)
    if args.max_episodes is not None:
        parquet_files = parquet_files[:args.max_episodes]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.input}")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")

    output = resolve_hdf5_output_path(args.output)
    ensure_dir(output.parent)
    compression = None if args.compression == "none" else args.compression

    print(f"Found {len(parquet_files)} parquet episode(s).")
    print(f"Output: {output}")
    print(
        "Action: relative trajectory, stored A_rel with "
        "A_abs = T_t @ A_rel"
    )
    print(
        f"Eye image: key={args.eye_image_key or f'{args.eye_camera}/color'}, "
        f"layout=CHW, size=3x{args.image_size}x{args.image_size}"
    )
    print(
        f"Agentview image: key={args.agentview_image_key or f'{args.agentview_camera}/color'}, "
        f"layout=CHW, size=3x{args.image_size}x{args.image_size}"
    )

    total_samples = 0
    written = 0
    with h5py.File(output, "w") as h5:
        data_group = h5.create_group("data")
        h5.attrs["format"] = "sym_realbot_image_rel_traj"
        h5.attrs["action_space"] = "relative_trajectory"
        h5.attrs["action_formula"] = "A_abs = T_t @ A_rel; stored A_rel = inv(T_t) @ A_abs"
        h5.attrs["eye_image_input_color_order"] = args.eye_image_color_order
        h5.attrs["eye_image_stored_color_order"] = "rgb"
        h5.attrs["eye_image_layout"] = "chw"
        h5.attrs["eye_image_size"] = int(args.image_size)
        h5.attrs["agentview_image_input_color_order"] = args.agentview_image_color_order
        h5.attrs["agentview_image_stored_color_order"] = "rgb"
        h5.attrs["agentview_image_layout"] = "chw"
        h5.attrs["agentview_image_size"] = int(args.image_size)
        h5.attrs["has_point_cloud"] = False
        h5.attrs["has_voxels"] = False

        for parquet_path in parquet_files:
            try:
                episode = build_episode(parquet_path, args)
                if episode is None:
                    print(f"[Skip] {parquet_path.name}: not enough valid frames")
                    continue
                total_samples += write_demo(data_group, written, episode, parquet_path, compression)
                print(f"[Write] demo_{written}: {parquet_path.name}, steps={episode['actions'].shape[0]}")
                written += 1
            except Exception as exc:
                if not args.skip_errors:
                    raise
                print(f"[Skip] {parquet_path.name}: {exc}")

        data_group.attrs["total"] = int(total_samples)
        data_group.attrs["num_demos"] = int(written)

    print(f"Done. Wrote {written}/{len(parquet_files)} demos, total samples={total_samples}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
