#!/usr/bin/env python
import argparse
import copy
import json
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

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
import robomimic.utils.file_utils as FileUtils

from reverse_controller.collect_inverse_dataset_osc import make_env_fn
from reverse_controller.common import (
    build_state_features,
    get_demo_names,
    load_inverse_checkpoint,
    parse_csv,
    predict_command,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obs-keys", default=None)
    parser.add_argument("--joint-keys", default=None)
    parser.add_argument("--demo-start", type=int, default=None)
    parser.add_argument("--demo-end", type=int, default=None)
    parser.add_argument("--max-demos", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=28)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, model, normalizer = load_inverse_checkpoint(args.checkpoint, device=args.device)
    meta = payload.get("dataset_metadata", {})
    dataset_path = os.path.expanduser(args.dataset or meta["dataset"])
    obs_keys = parse_csv(args.obs_keys) if args.obs_keys else list(meta["obs_keys"])
    joint_keys = parse_csv(args.joint_keys) if args.joint_keys else list(meta.get("joint_keys", ["robot0_joint_pos"]))
    command_scale = np.asarray(meta.get("joint_delta_scale", [1.0] * meta.get("action_dim", 7)), dtype=np.float32)

    demo_start = args.demo_start
    demo_end = args.demo_end
    n_demos = int(meta.get("n_demos", 0))
    if demo_start is None:
        demo_start = 250 if n_demos >= 300 else 150
    if demo_end is None:
        demo_end = 300 if n_demos >= 300 else 200

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    env = AsyncVectorEnv([make_env_fn(env_meta, joint_keys)] * args.n_envs)

    demo_names = get_demo_names(dataset_path)[demo_start:demo_end]
    if args.max_demos is not None:
        demo_names = demo_names[:args.max_demos]

    desired_all = []
    actual_all = []
    command_all = []
    demo_idx_all = []
    timestep_all = []

    try:
        for demo_name in tqdm.tqdm(demo_names, desc="eval demos"):
            demo_idx = int(demo_name.split("_")[-1])
            with h5py.File(dataset_path, "r") as f:
                demo = f[f"data/{demo_name}"]
                states = np.asarray(demo["states"][:])
                obs = demo["obs"]
                next_obs = demo["next_obs"]
                n_steps = int(demo["actions"].shape[0])
                state_features = np.stack(
                    [build_state_features(obs, obs_keys, t) for t in range(n_steps)],
                    axis=0,
                ).astype(np.float32)
                desired = np.concatenate(
                    [
                        np.asarray(next_obs[key][:], dtype=np.float32)
                        - np.asarray(obs[key][:], dtype=np.float32)
                        for key in joint_keys
                    ],
                    axis=-1,
                ).astype(np.float32)

            command = predict_command(
                model=model,
                normalizer=normalizer,
                state=state_features,
                desired_delta=desired,
                command_scale=command_scale,
            ).astype(np.float32)

            actual = np.zeros_like(desired, dtype=np.float32)
            for start in range(0, n_steps, args.n_envs):
                end = min(start + args.n_envs, n_steps)
                args_list = [
                    (states[i], command[i], joint_keys)
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
        "checkpoint": args.checkpoint,
        "dataset": dataset_path,
        "demo_start": int(demo_start),
        "demo_end": int(demo_end),
        "n_demos": int(len(demo_names)),
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

    n_joints = desired.shape[1]
    ncols = min(4, n_joints)
    nrows = int(np.ceil(n_joints / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for j in range(n_joints):
        ax = axes[j]
        ax.hexbin(desired[:, j], actual[:, j], gridsize=70, bins="log", mincnt=1)
        lo = float(min(np.min(desired[:, j]), np.min(actual[:, j])))
        hi = float(max(np.max(desired[:, j]), np.max(actual[:, j])))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"joint {j}")
        ax.set_xlabel("desired delta")
        ax.set_ylabel("actual delta")
    for j in range(n_joints, len(axes)):
        axes[j].axis("off")
    fig.savefig(output_dir / "desired_vs_actual.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
