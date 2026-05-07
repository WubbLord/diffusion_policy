import argparse
import copy
import csv
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

import robomimic.utils.file_utils as FileUtils

from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.env_runner.robomimic_joint_lowdim_runner import (
    _make_joint_position_controller_configs,
)
from reverse_controller.collect_inverse_dataset import build_controller_action
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


def success_value(env):
    try:
        value = env.is_success()
    except Exception:
        return False
    if isinstance(value, dict):
        if "task" in value:
            return bool(value["task"])
        return bool(any(value.values()))
    return bool(value)


def safe_norm(value):
    value = np.asarray(value, dtype=np.float32)
    return float(np.linalg.norm(value))


def metric_row_from_steps(demo_idx, demo_name, rewards, success_flags, q_error, eef_pos_error, object_error):
    rewards = np.asarray(rewards, dtype=np.float32)
    success_flags = np.asarray(success_flags, dtype=bool)
    q_error = np.asarray(q_error, dtype=np.float32)
    row = {
        "demo_idx": int(demo_idx),
        "demo_name": demo_name,
        "n_steps": int(len(rewards)),
        "max_reward": float(np.max(rewards)) if len(rewards) else 0.0,
        "success": bool(np.max(success_flags)) if len(success_flags) else False,
        "first_success_step": int(np.argmax(success_flags)) if np.any(success_flags) else -1,
        "mean_q_l2_error": float(np.mean(np.linalg.norm(q_error, axis=-1))),
        "max_q_l2_error": float(np.max(np.linalg.norm(q_error, axis=-1))),
        "final_q_l2_error": safe_norm(q_error[-1]),
        "mean_q_abs_error": float(np.mean(np.abs(q_error))),
        "final_q_abs_error": float(np.mean(np.abs(q_error[-1]))),
    }
    if eef_pos_error:
        eef_pos_error = np.asarray(eef_pos_error, dtype=np.float32)
        row.update({
            "mean_eef_pos_l2_error": float(np.mean(np.linalg.norm(eef_pos_error, axis=-1))),
            "final_eef_pos_l2_error": safe_norm(eef_pos_error[-1]),
        })
    if object_error:
        object_error = np.asarray(object_error, dtype=np.float32)
        row.update({
            "mean_object_l2_error": float(np.mean(np.linalg.norm(object_error, axis=-1))),
            "final_object_l2_error": safe_norm(object_error[-1]),
        })
    return row


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_state_features_from_obs(obs, obs_keys):
    missing = [key for key in obs_keys if key not in obs]
    if missing:
        raise KeyError(f"Current env observation is missing keys required by f: {missing}")
    return np.concatenate(
        [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in obs_keys],
        axis=0,
    )


def make_figures(output_dir, arrays, summary):
    output_dir = pathlib.Path(output_dir)
    q_error = arrays["q_error"]
    actual_delta = arrays["actual_delta"]
    desired_delta = arrays["desired_delta"]
    command = arrays["command"]
    step = arrays["timestep"]
    demo_idx = arrays["demo_idx"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    for j in range(desired_delta.shape[1]):
        ax = axes[j]
        ax.hexbin(desired_delta[:, j], actual_delta[:, j], gridsize=70, bins="log", mincnt=1)
        lo = float(min(np.percentile(desired_delta[:, j], 0.2), np.percentile(actual_delta[:, j], 0.2)))
        hi = float(max(np.percentile(desired_delta[:, j], 99.8), np.percentile(actual_delta[:, j], 99.8)))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"joint {j}")
        ax.set_xlabel("demo desired delta")
        ax.set_ylabel("replay actual delta")
    for j in range(desired_delta.shape[1], len(axes)):
        axes[j].axis("off")
    fig.suptitle("Oracle replay one-step deltas during full rollout", fontsize=14)
    fig.savefig(output_dir / "desired_vs_actual_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    unique_demos = np.unique(demo_idx)
    for demo in unique_demos[:10]:
        mask = demo_idx == demo
        ax.plot(step[mask], np.linalg.norm(q_error[mask], axis=-1), alpha=0.75, label=f"demo_{demo}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("q drift ||q_replay - q_demo||2")
    ax.set_title("Joint drift over oracle replay, first 10 demos")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "q_drift_examples.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(np.arange(7), np.mean(np.abs(actual_delta - desired_delta), axis=0))
    axes[0].set_xlabel("joint")
    axes[0].set_ylabel("mean |actual_delta - desired_delta|")
    axes[0].set_title("Per-step delta tracking error")
    axes[1].bar(np.arange(7), np.mean(np.abs(q_error), axis=0))
    axes[1].set_xlabel("joint")
    axes[1].set_ylabel("mean |q replay - q demo|")
    axes[1].set_title("Replay state drift")
    fig.savefig(output_dir / "per_joint_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(np.max(np.abs(command), axis=-1), bins=80)
    ax.set_xlabel("max per-joint |command| (rad)")
    ax.set_ylabel("count")
    ax.set_title("Pseudo-command magnitudes")
    fig.savefig(output_dir / "command_magnitude_hist.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["f", "raw"], default="f")
    parser.add_argument("--obs-keys", default=None)
    parser.add_argument("--joint-key", default=None)
    parser.add_argument("--gripper-action-indices", default=None)
    parser.add_argument("--joint-delta-scale", default=None)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-end", type=int, default=None)
    parser.add_argument("--max-demos", type=int, default=20)
    parser.add_argument(
        "--state-source",
        choices=["current", "logged"],
        default="current",
        help="Use the live replay observation or the logged demo observation as f's state input.",
    )
    parser.add_argument(
        "--desired-source",
        choices=["current_to_demo_next", "logged_delta"],
        default="current_to_demo_next",
        help=(
            "current_to_demo_next uses demo_q[t+1] - current_replay_q; "
            "logged_delta uses demo_q[t+1] - demo_q[t]."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "f" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --mode=f")

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = None
    normalizer = None
    checkpoint_meta = {}
    if args.mode == "f":
        payload, model, normalizer = load_inverse_checkpoint(args.checkpoint, device=args.device)
        checkpoint_meta = payload["dataset_metadata"]
    else:
        payload = None

    dataset_path = os.path.expanduser(args.dataset)
    obs_keys = (
        parse_csv(args.obs_keys)
        if args.obs_keys
        else list(checkpoint_meta.get("obs_keys", [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
        ]))
    )
    joint_key = args.joint_key or checkpoint_meta.get("joint_key", "robot0_joint_pos")

    with h5py.File(dataset_path, "r") as f:
        action_dim = f["data/demo_0/actions"].shape[-1]
        joint_dim = f["data/demo_0/obs"][joint_key].shape[-1]
    if args.gripper_action_indices:
        gripper_action_indices = parse_int_csv(args.gripper_action_indices)
    else:
        gripper_action_indices = list(checkpoint_meta.get("gripper_action_indices", [-1]))
    gripper_action_indices = [
        idx if idx >= 0 else action_dim + idx
        for idx in gripper_action_indices
    ]

    if args.joint_delta_scale:
        command_scale = expand_scale(parse_float_csv(args.joint_delta_scale), joint_dim)
    else:
        command_scale = np.asarray(
            checkpoint_meta.get("joint_delta_scale", [0.25] * joint_dim),
            dtype=np.float32,
        )

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    env_meta["env_kwargs"]["controller_configs"] = _make_joint_position_controller_configs(
        [command_scale]
    )
    replay_obs_keys = sorted(set(obs_keys + [joint_key, "robot0_eef_pos", "object"]))
    env = create_env(env_meta=env_meta, obs_keys=replay_obs_keys)

    demo_names = get_demo_names(dataset_path)
    demo_names = demo_names[args.demo_start:args.demo_end]
    if args.max_demos is not None:
        demo_names = demo_names[:args.max_demos]

    metadata = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "dataset": dataset_path,
        "obs_keys": obs_keys,
        "joint_key": joint_key,
        "joint_delta_scale": command_scale.tolist(),
        "gripper_action_indices": gripper_action_indices,
        "demo_start": args.demo_start,
        "demo_end": args.demo_end,
        "max_demos": args.max_demos,
        "state_source": args.state_source,
        "desired_source": args.desired_source,
        "demo_names": demo_names,
        "note": (
            "For state_source=current, f is evaluated on the live replay observation at each step. "
            "For desired_source=current_to_demo_next, desired_delta is demo_q[t+1] - current_replay_q, "
            "and delta tracking metrics compare that desired_delta to the realized controller delta."
        ),
    }
    save_json(output_dir / "metadata.json", metadata)

    records = {
        "demo_idx": [],
        "timestep": [],
        "desired_delta": [],
        "demo_logged_delta": [],
        "command": [],
        "actual_delta": [],
        "q_error": [],
        "reward": [],
        "success": [],
    }
    if "robot0_eef_pos" in replay_obs_keys:
        records["eef_pos_error"] = []
    if "object" in replay_obs_keys:
        records["object_error"] = []

    per_demo = []
    try:
        for demo_name in tqdm.tqdm(demo_names, desc="oracle replay demos"):
            demo_idx = int(demo_name.split("_")[-1])
            with h5py.File(dataset_path, "r") as f:
                demo = f[f"data/{demo_name}"]
                states = np.asarray(demo["states"][:])
                actions = np.asarray(demo["actions"][:], dtype=np.float32)
                obs = demo["obs"]
                next_obs = demo["next_obs"]
                n_steps = actions.shape[0]
                logged_state_features = np.stack(
                    [build_state_features(obs, obs_keys, t) for t in range(n_steps)],
                    axis=0,
                ).astype(np.float32)
                demo_next_q = np.asarray(next_obs[joint_key][:], dtype=np.float32)
                logged_delta = (
                    np.asarray(next_obs[joint_key][:], dtype=np.float32)
                    - np.asarray(obs[joint_key][:], dtype=np.float32)
                )
                demo_next_eef = (
                    np.asarray(next_obs["robot0_eef_pos"][:], dtype=np.float32)
                    if "robot0_eef_pos" in next_obs else None
                )
                demo_next_object = (
                    np.asarray(next_obs["object"][:], dtype=np.float32)
                    if "object" in next_obs else None
                )

            env.reset_to({"states": states[0]})
            rewards = []
            success_flags = []
            q_errors = []
            eef_errors = []
            object_errors = []
            current_obs = env.get_observation()
            prev_q = current_obs[joint_key].astype(np.float32).copy()

            for t in range(n_steps):
                if args.state_source == "current":
                    state_feature = build_state_features_from_obs(current_obs, obs_keys)
                else:
                    state_feature = logged_state_features[t]

                if args.desired_source == "current_to_demo_next":
                    desired_delta = demo_next_q[t] - prev_q
                else:
                    desired_delta = logged_delta[t]

                if args.mode == "f":
                    command = predict_command(
                        model=model,
                        normalizer=normalizer,
                        state=state_feature[None],
                        desired_delta=desired_delta[None],
                        command_scale=command_scale,
                    )[0].astype(np.float32)
                else:
                    command = np.clip(desired_delta, -command_scale, command_scale).astype(np.float32)

                env_action = build_controller_action(
                    command,
                    actions[t, gripper_action_indices],
                    command_scale,
                )
                current_obs, reward, done, info = env.step(env_action)
                q_after = current_obs[joint_key].astype(np.float32).copy()
                actual_delta = q_after - prev_q
                prev_q = q_after

                q_error = q_after - demo_next_q[t]
                q_errors.append(q_error)
                if demo_next_eef is not None and "robot0_eef_pos" in current_obs:
                    eef_errors.append(current_obs["robot0_eef_pos"].astype(np.float32) - demo_next_eef[t])
                if demo_next_object is not None and "object" in current_obs:
                    object_errors.append(current_obs["object"].astype(np.float32) - demo_next_object[t])

                success = success_value(env)
                rewards.append(float(reward))
                success_flags.append(success)

                records["demo_idx"].append(demo_idx)
                records["timestep"].append(t)
                records["desired_delta"].append(desired_delta)
                records["demo_logged_delta"].append(logged_delta[t])
                records["command"].append(command)
                records["actual_delta"].append(actual_delta)
                records["q_error"].append(q_error)
                records["reward"].append(float(reward))
                records["success"].append(success)
                if "eef_pos_error" in records and eef_errors:
                    records["eef_pos_error"].append(eef_errors[-1])
                if "object_error" in records and object_errors:
                    records["object_error"].append(object_errors[-1])

            per_demo.append(metric_row_from_steps(
                demo_idx=demo_idx,
                demo_name=demo_name,
                rewards=rewards,
                success_flags=success_flags,
                q_error=q_errors,
                eef_pos_error=eef_errors,
                object_error=object_errors,
            ))
    finally:
        close_fn = getattr(env, "close", None)
        if close_fn is not None:
            close_fn()

    arrays = {}
    for key, values in records.items():
        if key in {"demo_idx", "timestep"}:
            arrays[key] = np.asarray(values, dtype=np.int32)
        elif key == "success":
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(output_dir / "oracle_replay.npz", **arrays)
    write_csv(output_dir / "per_demo_metrics.csv", per_demo)
    save_json(output_dir / "per_demo_metrics.json", per_demo)

    success_rate = float(np.mean([row["success"] for row in per_demo])) if per_demo else 0.0
    summary = {
        "n_demos": len(per_demo),
        "n_steps": int(len(arrays["timestep"])),
        "success_rate": success_rate,
        "mean_max_reward": float(np.mean([row["max_reward"] for row in per_demo])) if per_demo else 0.0,
        "mean_final_q_l2_error": float(np.mean([row["final_q_l2_error"] for row in per_demo])) if per_demo else 0.0,
        "mean_final_q_abs_error": float(np.mean([row["final_q_abs_error"] for row in per_demo])) if per_demo else 0.0,
        "mean_q_l2_error": float(np.mean([row["mean_q_l2_error"] for row in per_demo])) if per_demo else 0.0,
        "mean_q_abs_error": float(np.mean([row["mean_q_abs_error"] for row in per_demo])) if per_demo else 0.0,
        "delta_tracking_mae": float(np.mean(np.abs(arrays["actual_delta"] - arrays["desired_delta"]))),
        "delta_tracking_rmse": float(np.sqrt(np.mean((arrays["actual_delta"] - arrays["desired_delta"]) ** 2))),
    }
    if "eef_pos_error" in arrays and len(arrays["eef_pos_error"]):
        summary["mean_final_eef_pos_l2_error"] = float(np.mean([
            row.get("final_eef_pos_l2_error", 0.0) for row in per_demo
        ]))
        summary["mean_eef_pos_l2_error"] = float(np.mean([
            row.get("mean_eef_pos_l2_error", 0.0) for row in per_demo
        ]))
    if "object_error" in arrays and len(arrays["object_error"]):
        summary["mean_final_object_l2_error"] = float(np.mean([
            row.get("final_object_l2_error", 0.0) for row in per_demo
        ]))
        summary["mean_object_l2_error"] = float(np.mean([
            row.get("mean_object_l2_error", 0.0) for row in per_demo
        ]))
    save_json(output_dir / "summary.json", summary)
    make_figures(output_dir, arrays, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
