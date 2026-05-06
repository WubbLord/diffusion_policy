import argparse
import copy
import os
import pathlib
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import tqdm

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffusion_policy.env_runner.robomimic_joint_lowdim_runner import (
    _make_joint_position_controller_configs,
)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
import robomimic.utils.file_utils as FileUtils

from reverse_controller.collect_inverse_dataset import (
    build_controller_action,
    make_env_fn,
)
from reverse_controller.common import (
    build_state_features,
    expand_scale,
    get_demo_names,
    load_inverse_checkpoint,
    parse_csv,
    parse_float_csv,
    parse_int_csv,
    predict_command,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obs-keys", default=None)
    parser.add_argument("--joint-key", default=None)
    parser.add_argument("--gripper-action-indices", default=None)
    parser.add_argument("--joint-delta-scale", default=None)
    parser.add_argument("--n-envs", type=int, default=28)
    parser.add_argument("--max-demos", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, model, normalizer = load_inverse_checkpoint(args.checkpoint, device=args.device)
    meta = payload["dataset_metadata"]
    dataset_path = os.path.expanduser(args.dataset or meta["dataset"])
    obs_keys = parse_csv(args.obs_keys) if args.obs_keys else list(meta["obs_keys"])
    joint_key = args.joint_key or meta["joint_key"]

    with h5py.File(dataset_path, "r") as f:
        action_dim = f["data/demo_0/actions"].shape[-1]
        joint_dim = f["data/demo_0/obs"][joint_key].shape[-1]
    if args.gripper_action_indices:
        gripper_action_indices = parse_int_csv(args.gripper_action_indices)
    else:
        gripper_action_indices = list(meta["gripper_action_indices"])
    gripper_action_indices = [
        idx if idx >= 0 else action_dim + idx
        for idx in gripper_action_indices
    ]

    if args.joint_delta_scale:
        command_scale = expand_scale(parse_float_csv(args.joint_delta_scale), joint_dim)
    else:
        command_scale = np.asarray(meta["joint_delta_scale"], dtype=np.float32)

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    env_meta["env_kwargs"]["controller_configs"] = _make_joint_position_controller_configs(
        [command_scale]
    )
    env_fn = make_env_fn(env_meta, joint_key)
    env = AsyncVectorEnv([env_fn] * args.n_envs)

    demo_names = get_demo_names(dataset_path)
    if args.max_demos is not None:
        demo_names = demo_names[:args.max_demos]

    desired_all = []
    actual_all = []
    command_all = []
    demo_idx_all = []
    timestep_all = []

    try:
        for demo_idx, demo_name in enumerate(tqdm.tqdm(demo_names, desc="eval demos")):
            with h5py.File(dataset_path, "r") as f:
                demo = f[f"data/{demo_name}"]
                states = np.asarray(demo["states"][:])
                actions = np.asarray(demo["actions"][:], dtype=np.float32)
                obs = demo["obs"]
                next_obs = demo["next_obs"]
                n_steps = actions.shape[0]

                state_features = np.stack(
                    [build_state_features(obs, obs_keys, t) for t in range(n_steps)],
                    axis=0,
                ).astype(np.float32)
                desired = (
                    np.asarray(next_obs[joint_key][:], dtype=np.float32)
                    - np.asarray(obs[joint_key][:], dtype=np.float32)
                )

            command = predict_command(
                model=model,
                normalizer=normalizer,
                state=state_features,
                desired_delta=desired,
                command_scale=command_scale,
            ).astype(np.float32)

            actual = np.zeros_like(command)
            for start in range(0, n_steps, args.n_envs):
                end = min(start + args.n_envs, n_steps)
                env_actions = [
                    build_controller_action(
                        command[i],
                        actions[i, gripper_action_indices],
                        command_scale,
                    )
                    for i in range(start, end)
                ]
                args_list = [
                    (states[i], env_actions[i - start], joint_key)
                    for i in range(start, end)
                ]
                if len(args_list) < args.n_envs:
                    args_list.extend([args_list[0]] * (args.n_envs - len(args_list)))
                result = env.call_each("probe", args_list=args_list)
                actual[start:end] = np.asarray(result[:end - start], dtype=np.float32)

            desired_all.append(desired)
            actual_all.append(actual)
            command_all.append(command)
            demo_idx_all.append(np.full((n_steps,), demo_idx, dtype=np.int32))
            timestep_all.append(np.arange(n_steps, dtype=np.int32))
    finally:
        env.close()

    desired = np.concatenate(desired_all, axis=0)
    actual = np.concatenate(actual_all, axis=0)
    command = np.concatenate(command_all, axis=0)
    demo_idx = np.concatenate(demo_idx_all, axis=0)
    timestep = np.concatenate(timestep_all, axis=0)
    err = actual - desired

    np.savez_compressed(
        output_dir / "one_step_eval.npz",
        desired_delta=desired,
        actual_delta=actual,
        command=command,
        demo_idx=demo_idx,
        timestep=timestep,
    )
    summary = {
        "n_samples": int(desired.shape[0]),
        "mean_abs_desired": float(np.mean(np.abs(desired))),
        "mean_abs_actual": float(np.mean(np.abs(actual))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "rmse_error": float(np.sqrt(np.mean(err ** 2))),
        "per_joint": [],
    }
    for j in range(desired.shape[1]):
        summary["per_joint"].append({
            "joint": j,
            "mean_abs_desired": float(np.mean(np.abs(desired[:, j]))),
            "mean_abs_actual": float(np.mean(np.abs(actual[:, j]))),
            "mean_abs_error": float(np.mean(np.abs(err[:, j]))),
            "rmse_error": float(np.sqrt(np.mean(err[:, j] ** 2))),
        })
    save_json(output_dir / "summary.json", summary)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    for j in range(desired.shape[1]):
        ax = axes[j]
        ax.hexbin(desired[:, j], actual[:, j], gridsize=70, bins="log", mincnt=1)
        lo = float(min(np.min(desired[:, j]), np.min(actual[:, j])))
        hi = float(max(np.max(desired[:, j]), np.max(actual[:, j])))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"joint {j}")
        ax.set_xlabel("demo desired joint delta")
        ax.set_ylabel("actual delta after f command")
    for j in range(desired.shape[1], len(axes)):
        axes[j].axis("off")
    fig.savefig(output_dir / "desired_vs_actual.png", dpi=180)
    plt.close(fig)

    print(summary, flush=True)


if __name__ == "__main__":
    main()
