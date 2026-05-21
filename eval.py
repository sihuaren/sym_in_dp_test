#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import pathlib
import random
import sys
from datetime import datetime

# Headless robosuite / mujoco rendering, matching the training command.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)


MAX_STEPS = {
    "stack_d0": 400,
    "stack_d1": 400,
    "stack_three_d1": 400,
    "square_d2": 400,
    "threading_d0": 400,
    "threading_d2": 400,
    "coffee_d2": 400,
    "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500,
    "mug_cleanup_d1": 500,
    "kitchen_d1": 800,
    "nut_assembly_d0": 500,
    "pick_place_d0": 1000,
    "coffee_preparation_d1": 800,
    "tool_hang": 700,
    "can": 400,
    "lift": 400,
    "square": 400,
    "three_piece_assembly_d0": 500,
}


def get_ws_x_center(task_name):
    if task_name.startswith("kitchen_") or task_name.startswith("hammer_cleanup_"):
        return -0.2
    return 0.0


def get_ws_y_center(task_name):
    return 0.0


def get_ws_z_center(task_name):
    if task_name.startswith("kitchen_") or task_name.startswith("hammer_cleanup_"):
        return 0.9
    return 0.8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a sym_in_dp diffusion checkpoint on MimicGen / robomimic."
    )
    parser.add_argument(
        "--ckpt",
        default="./data/outputs/2026.05.18/13.43.59_diff_c_pre_rel_traj_coffee_d2/checkpoints/epoch=0100-test_mean_score=0.400.ckpt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--task-name",
        default="coffee_d2",
        help="MimicGen task name. Defaults to coffee_d2.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Dataset used to construct the robomimic env. Defaults to "
        "data/robomimic/datasets/<task>/<task>_fisheye_abs.hdf5.",
    )
    parser.add_argument(
        "--robomimic-path",
        default="/data1/user/rensihua/robomimic",
        help="Path to the robomimic project checkout.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for videos and metrics. Defaults to data/eval_outputs/<timestamp>_<task>.",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Parallel env count. Defaults to cfg.task.env_runner.n_envs, capped by num episodes.",
    )
    parser.add_argument("--device", default=None, help="cuda:0, cpu, etc. Defaults to training config.")
    parser.add_argument(
        "--policy-key",
        choices=["auto", "ema_model", "model"],
        default="auto",
        help="Which checkpoint state_dict to evaluate.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument(
        "--use-sync-env",
        action="store_true",
        help="Use single-process vector env. Helpful for debugging multiprocessing issues.",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--crf", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Torch/numpy/random seed for policy sampling.")
    return parser.parse_args()


def setup_paths(root_dir, robomimic_path):
    root_dir = pathlib.Path(root_dir).resolve()
    os.chdir(root_dir)
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    robomimic_path = pathlib.Path(robomimic_path).expanduser()
    if robomimic_path.exists() and str(robomimic_path) not in sys.path:
        sys.path.insert(0, str(robomimic_path))
    elif not robomimic_path.exists():
        print(f"[warn] robomimic path does not exist: {robomimic_path}")


def register_omegaconf_resolvers(OmegaConf):
    OmegaConf.register_new_resolver("get_max_steps", lambda x: MAX_STEPS[x], replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
    OmegaConf.register_new_resolver("get_ws_z_center", get_ws_z_center, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def torch_load_checkpoint(torch, dill, ckpt_path):
    with open(ckpt_path, "rb") as f:
        try:
            return torch.load(
                f,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
        except TypeError:
            f.seek(0)
            return torch.load(f, map_location="cpu", pickle_module=dill)


def to_plain_container(OmegaConf, cfg_node):
    return OmegaConf.to_container(cfg_node, resolve=True)


def create_robomimic_env(EnvUtils, ObsUtils, env_meta, shape_meta, enable_render=True):
    import collections

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )


def undo_transform_action(rotation_transformer, action):
    import numpy as np

    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    untransformed_action = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        untransformed_action = untransformed_action.reshape(*raw_shape[:-1], 14)

    return untransformed_action


def make_final_video_path(video_dir, episode, seed, success, max_reward):
    status = "success" if success else "fail"
    return video_dir.joinpath(
        f"episode_{episode:03d}_seed_{seed}_{status}_score_{max_reward:.3f}.mp4"
    )


def safe_rename_video(src, dst):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    return str(dst)


def resize_video_frame(frame, size=(224, 224)):
    if frame.shape[:2] == size:
        return frame

    try:
        import cv2

        resized = cv2.resize(
            frame,
            (size[1], size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    except ImportError:
        import numpy as np
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((size[1], size[0])))

    return resized.astype(frame.dtype, copy=False)


def evaluate(
    *,
    cfg,
    OmegaConf,
    FileUtils,
    EnvUtils,
    ObsUtils,
    AsyncVectorEnv,
    SyncVectorEnv,
    MultiStepWrapper,
    VideoRecordingWrapper,
    VideoRecorder,
    RobomimicImageWrapper,
    RotationTransformer,
    dict_apply,
    dill,
    torch,
    np,
    policy,
    dataset_path,
    output_dir,
    num_episodes,
    start_seed,
    n_envs,
    success_threshold,
    use_sync_env,
    fps,
    crf,
):
    runner_cfg = cfg.task.env_runner
    shape_meta = to_plain_container(OmegaConf, runner_cfg.shape_meta)
    render_obs_key = str(runner_cfg.get("render_obs_key", "robot0_eye_in_hand_image"))
    n_obs_steps = int(runner_cfg.get("n_obs_steps", cfg.n_obs_steps))
    n_action_steps = int(runner_cfg.get("n_action_steps", cfg.n_action_steps))
    max_steps = int(runner_cfg.get("max_steps", MAX_STEPS[cfg.task_name]))
    fps = int(fps if fps is not None else runner_cfg.get("fps", 10))
    crf = int(crf if crf is not None else runner_cfg.get("crf", 22))
    abs_action = bool(runner_cfg.get("abs_action", True))

    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)

    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env_meta["env_kwargs"]["use_object_obs"] = False

    rotation_transformer = None
    if abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    video_dir = pathlib.Path(output_dir).joinpath("videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    temp_video_dir = video_dir.joinpath("_tmp")
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    def env_fn(enable_render=True):
        robomimic_env = create_robomimic_env(
            EnvUtils=EnvUtils,
            ObsUtils=ObsUtils,
            env_meta=env_meta,
            shape_meta=shape_meta,
            enable_render=enable_render,
        )
        robomimic_env.env.hard_reset = False

        wrapped_env = RobomimicImageWrapper(
            env=robomimic_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=render_obs_key,
        )
        raw_render = wrapped_env.render

        def render_224(mode="rgb_array", **kwargs):
            return resize_video_frame(raw_render(mode=mode, **kwargs), size=(224, 224))

        wrapped_env.render = render_224

        return MultiStepWrapper(
            VideoRecordingWrapper(
                wrapped_env,
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=crf,
                    thread_type="FRAME",
                    thread_count=1,
                ),
                file_path=None,
                steps_per_render=steps_per_render,
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )

    def dummy_env_fn():
        return env_fn(enable_render=False)

    if n_envs is None:
        n_envs = int(runner_cfg.get("n_envs", num_episodes))
    n_envs = max(1, min(int(n_envs), int(num_episodes)))
    vector_env_cls = SyncVectorEnv if use_sync_env else AsyncVectorEnv
    if use_sync_env:
        env = vector_env_cls([env_fn] * n_envs)
    else:
        env = vector_env_cls([env_fn] * n_envs, dummy_env_fn=dummy_env_fn)

    device = policy.device
    env_name = env_meta["env_name"]
    episode_specs = []
    for episode in range(num_episodes):
        seed = start_seed + episode
        pending_video_path = temp_video_dir.joinpath(
            f"episode_{episode:03d}_seed_{seed}_pending.mp4"
        )
        episode_specs.append(
            {
                "episode": episode,
                "seed": seed,
                "pending_video_path": str(pending_video_path),
            }
        )
    
    def make_init_fn(seed, pending_video_path=None):
        def init_fn(env):
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = str(pending_video_path) if pending_video_path is not None else None

            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None
            env.seed(seed)

        return dill.dumps(init_fn)

    results = []
    try:
        n_chunks = math.ceil(num_episodes / n_envs)
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(num_episodes, start + n_envs)
            active_specs = episode_specs[start:end]
            init_fn_dills = [
                make_init_fn(x["seed"], x["pending_video_path"]) for x in active_specs
            ]

            # Padded envs must not write to the same video path as a real episode.
            while len(init_fn_dills) < n_envs:
                init_fn_dills.append(make_init_fn(start_seed, None))

            env.call_each("run_dill_function", args_list=[(x,) for x in init_fn_dills])
            obs = env.reset()
            policy.reset()

            import tqdm

            pbar = tqdm.tqdm(
                total=max_steps,
                desc=f"Eval {env_name} {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=float(runner_cfg.get("tqdm_interval_sec", 1.0)),
            )

            done = np.zeros((n_envs,), dtype=np.bool_)
            while not np.all(done):
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError(f"NaN or Inf action encountered: {action}")

                env_action = action
                if abs_action:
                    env_action = undo_transform_action(rotation_transformer, action)

                obs, reward, done, info = env.step(env_action)
                pbar.update(action.shape[1])
            pbar.close()

            video_paths = list(env.render())[: len(active_specs)]
            reward_lists = list(env.call("get_attr", "reward"))[: len(active_specs)]

            for spec, pending_video_path, reward_list in zip(
                active_specs, video_paths, reward_lists
            ):
                reward_array = np.asarray(reward_list, dtype=np.float32)
                max_reward = float(np.max(reward_array)) if reward_array.size else 0.0
                success = bool(max_reward >= success_threshold)
                final_video_path = make_final_video_path(
                    video_dir=video_dir,
                    episode=spec["episode"],
                    seed=spec["seed"],
                    success=success,
                    max_reward=max_reward,
                )
                renamed_video_path = safe_rename_video(pending_video_path, final_video_path)
                results.append(
                    {
                        "episode": spec["episode"],
                        "seed": spec["seed"],
                        "success": int(success),
                        "max_reward": max_reward,
                        "num_env_steps": int(len(reward_list)),
                        "video_path": renamed_video_path,
                    }
                )
                print(
                    f"episode={spec['episode']:03d} seed={spec['seed']} "
                    f"success={int(success)} max_reward={max_reward:.3f} "
                    f"video={renamed_video_path}"
                )

        success_rate = float(np.mean([x["success"] for x in results])) if results else 0.0
        return results, success_rate
    finally:
        env.close()


def main():
    args = parse_args()
    root_dir = pathlib.Path(__file__).resolve().parent
    setup_paths(root_dir, args.robomimic_path)

    import dill
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import sys
    sys.path.insert(0, '/data1/user/rensihua/robomimic')

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    from sym_in_dp.common.pytorch_util import dict_apply
    from sym_in_dp.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
    from sym_in_dp.gym_util.async_vector_env import AsyncVectorEnv
    from sym_in_dp.gym_util.multistep_wrapper import MultiStepWrapper
    from sym_in_dp.gym_util.sync_vector_env import SyncVectorEnv
    from sym_in_dp.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
    from sym_in_dp.model.common.rotation_transformer import RotationTransformer

    register_omegaconf_resolvers(OmegaConf)

    ckpt_path = pathlib.Path(args.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = root_dir.joinpath(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch_load_checkpoint(torch, dill, ckpt_path)
    cfg = payload["cfg"]
    OmegaConf.set_struct(cfg, False)

    if args.task_name is not None:
        cfg.task_name = args.task_name

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = f"/hard_data/user_dataset/rensihua_dataset/sym_in_dp/data/robomimic/datasets/{cfg.task_name}/{cfg.task_name}_fisheye_abs.hdf5"
    cfg.dataset_path = dataset_path
    if "task" in cfg and "env_runner" in cfg.task:
        cfg.task.env_runner.dataset_path = dataset_path
        if cfg.task_name in MAX_STEPS:
            cfg.task.env_runner.max_steps = MAX_STEPS[cfg.task_name]

    device = args.device if args.device is not None else str(cfg.training.device)
    cfg.training.device = device
    OmegaConf.resolve(cfg)

    dataset_path = pathlib.Path(str(cfg.dataset_path)).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = root_dir.joinpath(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y.%m.%d/%H.%M.%S")
        output_dir = pathlib.Path("data/eval_outputs").joinpath(
            timestamp + f"_eval_{cfg.task_name}"
        )
    output_dir = pathlib.Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root_dir.joinpath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    if seed is None:
        seed = int(cfg.training.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Avoid a needless timm download: checkpoint weights overwrite initialization.
    if "obs_encoder" in cfg.policy and "pretrained" in cfg.policy.obs_encoder:
        cfg.policy.obs_encoder.pretrained = False

    policy = hydra.utils.instantiate(cfg.policy)
    state_key = args.policy_key
    if state_key == "auto":
        state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    if state_key not in payload["state_dicts"]:
        raise KeyError(
            f"State dict '{state_key}' not in checkpoint. Available: "
            f"{sorted(payload['state_dicts'].keys())}"
        )
    policy.load_state_dict(payload["state_dicts"][state_key], strict=True)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    policy.to(torch.device(device))
    policy.eval()

    start_seed = args.start_seed
    if start_seed is None:
        start_seed = int(cfg.task.env_runner.get("test_start_seed", 100000))

    print(f"checkpoint: {ckpt_path}")
    print(f"policy state: {state_key}")
    print(f"task: {cfg.task_name}")
    print(f"dataset: {dataset_path}")
    print(f"output_dir: {output_dir}")
    print(f"episodes: {args.num_episodes}, start_seed: {start_seed}")

    results, success_rate = evaluate(
        cfg=cfg,
        OmegaConf=OmegaConf,
        FileUtils=FileUtils,
        EnvUtils=EnvUtils,
        ObsUtils=ObsUtils,
        AsyncVectorEnv=AsyncVectorEnv,
        SyncVectorEnv=SyncVectorEnv,
        MultiStepWrapper=MultiStepWrapper,
        VideoRecordingWrapper=VideoRecordingWrapper,
        VideoRecorder=VideoRecorder,
        RobomimicImageWrapper=RobomimicImageWrapper,
        RotationTransformer=RotationTransformer,
        dict_apply=dict_apply,
        dill=dill,
        torch=torch,
        np=np,
        policy=policy,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_episodes=int(args.num_episodes),
        start_seed=start_seed,
        n_envs=args.n_envs,
        success_threshold=float(args.success_threshold),
        use_sync_env=bool(args.use_sync_env),
        fps=args.fps,
        crf=args.crf,
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "policy_state": state_key,
        "task_name": str(cfg.task_name),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "num_episodes": int(args.num_episodes),
        "start_seed": int(start_seed),
        "success_threshold": float(args.success_threshold),
        "successes": int(sum(x["success"] for x in results)),
        "success_rate": success_rate,
        "results": results,
    }

    summary_path = output_dir.joinpath("summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = output_dir.joinpath("results.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "seed",
                "success",
                "max_reward",
                "num_env_steps",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"success_rate: {success_rate:.4f} ({summary['successes']}/{args.num_episodes})")
    print(f"summary: {summary_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
