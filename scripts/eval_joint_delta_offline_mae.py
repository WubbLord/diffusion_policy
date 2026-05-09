#!/usr/bin/env python
import argparse
import json
import os
import pathlib
import sys

import dill
import h5py
import hydra
import numpy as np
import torch
import tqdm


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply


def save_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def build_obs(raw_obs, obs_keys, t):
    return np.concatenate(
        [np.asarray(raw_obs[key][t], dtype=np.float32).reshape(-1) for key in obs_keys],
        axis=0,
    )


def parse_int_csv(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def get_demo_indices(dataset_path, val_ratio, seed, override):
    with h5py.File(dataset_path, "r") as f:
        n_demos = len(f["data"])
    if override:
        return parse_int_csv(override)
    val_mask = get_val_mask(n_episodes=n_demos, val_ratio=val_ratio, seed=seed)
    return np.flatnonzero(val_mask).astype(int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-indices", default=None)
    parser.add_argument("--max-steps-per-demo", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = pathlib.Path(args.checkpoint)
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)

    if args.model == "ema" and cfg.training.use_ema:
        policy = workspace.ema_model
        model_name = "ema_model"
    else:
        policy = workspace.model
        model_name = "model"
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    dataset_cfg = cfg.task.dataset
    dataset_path = os.path.expanduser(str(dataset_cfg.dataset_path))
    obs_keys = list(dataset_cfg.obs_keys)
    joint_key = list(dataset_cfg.joint_pos_keys)[0]
    gripper_indices = [int(x) for x in list(dataset_cfg.gripper_action_indices)]
    val_ratio = float(dataset_cfg.val_ratio)
    split_seed = int(dataset_cfg.seed)
    demo_indices = get_demo_indices(
        dataset_path=dataset_path,
        val_ratio=val_ratio,
        seed=split_seed,
        override=args.demo_indices,
    )

    obs_batches = []
    target_joint_batches = []
    target_gripper_batches = []
    demo_idx_batches = []
    timestep_batches = []

    with h5py.File(dataset_path, "r") as f:
        action_dim = f["data/demo_0/actions"].shape[-1]
        gripper_indices = [
            idx if idx >= 0 else action_dim + idx
            for idx in gripper_indices
        ]

        for demo_idx in demo_indices:
            demo = f[f"data/demo_{demo_idx}"]
            raw_obs = demo["obs"]
            raw_actions = np.asarray(demo["actions"][:], dtype=np.float32)
            n_steps = min(raw_actions.shape[0], raw_obs[joint_key].shape[0]) - 1
            if args.max_steps_per_demo is not None:
                n_steps = min(n_steps, args.max_steps_per_demo)

            obs_seq = []
            target_joint = []
            target_gripper = []
            for t in range(n_steps):
                prev_t = max(t - 1, 0)
                obs_seq.append([
                    build_obs(raw_obs, obs_keys, prev_t),
                    build_obs(raw_obs, obs_keys, t),
                ])
                q = np.asarray(raw_obs[joint_key][t], dtype=np.float32)
                q_next = np.asarray(raw_obs[joint_key][t + 1], dtype=np.float32)
                target_joint.append(q_next - q)
                target_gripper.append(raw_actions[t, gripper_indices])

            if n_steps > 0:
                obs_batches.append(np.asarray(obs_seq, dtype=np.float32))
                target_joint_batches.append(np.asarray(target_joint, dtype=np.float32))
                target_gripper_batches.append(np.asarray(target_gripper, dtype=np.float32))
                demo_idx_batches.append(np.full((n_steps,), demo_idx, dtype=np.int32))
                timestep_batches.append(np.arange(n_steps, dtype=np.int32))

    obs = np.concatenate(obs_batches, axis=0)
    target_joint = np.concatenate(target_joint_batches, axis=0)
    target_gripper = np.concatenate(target_gripper_batches, axis=0)
    demo_idx = np.concatenate(demo_idx_batches, axis=0)
    timestep = np.concatenate(timestep_batches, axis=0)

    pred_action_parts = []
    with torch.no_grad():
        for start in tqdm.trange(0, obs.shape[0], args.batch_size, desc="offline action eval"):
            end = min(start + args.batch_size, obs.shape[0])
            batch = {
                "obs": torch.as_tensor(obs[start:end], dtype=torch.float32, device=device)
            }
            result = policy.predict_action(batch)
            pred_action_parts.append(result["action"][:, 0].detach().cpu().numpy())
            del result
            del batch
    pred_action = np.concatenate(pred_action_parts, axis=0).astype(np.float32)
    pred_joint = pred_action[:, :target_joint.shape[-1]]
    pred_gripper = pred_action[:, target_joint.shape[-1]:target_joint.shape[-1] + target_gripper.shape[-1]]

    joint_error = pred_joint - target_joint
    gripper_error = pred_gripper - target_gripper

    per_demo = []
    for demo in demo_indices:
        mask = demo_idx == demo
        per_demo.append({
            "demo_idx": int(demo),
            "n_steps": int(mask.sum()),
            "joint_mae": float(np.mean(np.abs(joint_error[mask]))),
            "joint_rmse": float(np.sqrt(np.mean(joint_error[mask] ** 2))),
            "mean_abs_target_joint_delta": float(np.mean(np.abs(target_joint[mask]))),
            "mean_abs_pred_joint_delta": float(np.mean(np.abs(pred_joint[mask]))),
        })

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(workspace.epoch),
        "checkpoint_global_step": int(workspace.global_step),
        "dataset": dataset_path,
        "demo_indices": [int(x) for x in demo_indices],
        "model": model_name,
        "seed": int(args.seed),
        "num_inference_steps": int(policy.num_inference_steps),
        "n_demos": int(len(demo_indices)),
        "n_steps": int(obs.shape[0]),
        "obs_keys": obs_keys,
        "joint_key": joint_key,
        "joint_mae": float(np.mean(np.abs(joint_error))),
        "joint_rmse": float(np.sqrt(np.mean(joint_error ** 2))),
        "mean_abs_target_joint_delta": float(np.mean(np.abs(target_joint))),
        "mean_abs_pred_joint_delta": float(np.mean(np.abs(pred_joint))),
        "per_joint_mae": np.mean(np.abs(joint_error), axis=0).astype(float).tolist(),
        "per_joint_rmse": np.sqrt(np.mean(joint_error ** 2, axis=0)).astype(float).tolist(),
        "gripper_mae": float(np.mean(np.abs(gripper_error))) if gripper_error.size else None,
        "gripper_rmse": float(np.sqrt(np.mean(gripper_error ** 2))) if gripper_error.size else None,
        "per_demo": per_demo,
    }

    np.savez_compressed(
        output_dir / "offline_joint_delta_mae.npz",
        pred_joint_delta=pred_joint,
        target_joint_delta=target_joint,
        pred_gripper=pred_gripper,
        target_gripper=target_gripper,
        demo_idx=demo_idx,
        timestep=timestep,
    )
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
