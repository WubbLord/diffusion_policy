"""OSC variant of Brian's eval_inverse_model.py — one-step open-loop adapter
quality eval.

For each held-out demo timestep:
  1. Reset sim to demo state
  2. Compute desired_Δq = q_demo[t+1] - q_demo[t]
  3. Predict OSC command via adapter: u = f(state, desired_Δq)
  4. Apply u, measure actual_Δq
  5. Record desired vs actual

Reports per-joint MAE, RMSE, and saves a hexbin scatter plot of
desired_Δq vs actual_Δq per joint (Brian's standard figure).

Usage:
    python eval_inverse_model_osc.py --task can_ph \\
        --adapter data/reverse_controller_osc_bq/can_ph/inverse_mlp/best.pt \\
        --output-dir <out>
"""
import argparse, json, os, copy, pathlib, sys
import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import tqdm

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
import robomimic.utils.file_utils as FileUtils
from reverse_controller.collect_inverse_dataset_osc import JointProbeEnvOSC, make_env_fn
from reverse_controller.common import (
    build_state_features, get_demo_names, load_inverse_checkpoint, parse_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--obs-keys", default="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos")
    ap.add_argument("--joint-keys", default="robot0_joint_pos")
    ap.add_argument("--n-envs", type=int, default=28)
    ap.add_argument("--demo-start", type=int, default=150)
    ap.add_argument("--demo-end", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dataset_path = f"data/robomimic/datasets/{args.task.replace('_', '/', 1)}/low_dim.hdf5"
    obs_keys = parse_csv(args.obs_keys)
    joint_keys = parse_csv(args.joint_keys)
    n_arms = len(joint_keys)

    payload, model, normalizer = load_inverse_checkpoint(args.adapter, device=args.device)

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    env_fn = make_env_fn(env_meta, joint_keys)
    env = AsyncVectorEnv([env_fn] * args.n_envs)

    demo_names = get_demo_names(dataset_path)[args.demo_start:args.demo_end]
    desired_all, actual_all, command_all = [], [], []

    try:
        for demo_name in tqdm.tqdm(demo_names, desc="eval demos"):
            with h5py.File(dataset_path, "r") as f:
                demo = f[f"data/{demo_name}"]
                states = np.asarray(demo["states"][:])
                actions = np.asarray(demo["actions"][:], dtype=np.float32)
                obs_group = demo["obs"]
                next_obs_group = demo["next_obs"]
                n_steps = actions.shape[0]

                state_features = np.stack(
                    [build_state_features(obs_group, obs_keys, t) for t in range(n_steps)],
                    axis=0).astype(np.float32)
                desired = np.concatenate([
                    np.asarray(next_obs_group[k][:], np.float32) - np.asarray(obs_group[k][:], np.float32)
                    for k in joint_keys
                ], axis=-1).astype(np.float32)

            # Predict command via adapter (batched)
            import torch
            with torch.no_grad():
                tx_input = np.concatenate([state_features, desired], axis=-1)
                tx = torch.as_tensor(tx_input, dtype=torch.float32, device=args.device)
                tx_norm = (tx - normalizer["input"]["mean"]) / normalizer["input"]["std"]
                pred = model(tx_norm)
                pred = pred * normalizer["command"]["std"] + normalizer["command"]["mean"]
            command = pred.detach().cpu().numpy().astype(np.float32)
            command = np.clip(command, -1.0, 1.0)  # OSC range

            # Probe: reset sim to each demo state, apply command, measure actual Δq
            actual = np.zeros_like(desired)
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
    finally:
        env.close()

    desired = np.concatenate(desired_all, axis=0)
    actual  = np.concatenate(actual_all,  axis=0)
    command = np.concatenate(command_all, axis=0)
    err = actual - desired

    np.savez_compressed(out / "one_step_eval.npz",
        desired_delta=desired, actual_delta=actual, command=command)

    summary = {
        "task": args.task,
        "adapter": args.adapter,
        "n_samples": int(desired.shape[0]),
        "demo_range": [args.demo_start, args.demo_end],
        "mean_abs_desired": float(np.mean(np.abs(desired))),
        "mean_abs_actual":  float(np.mean(np.abs(actual))),
        "mean_abs_error":   float(np.mean(np.abs(err))),
        "rmse_error":       float(np.sqrt(np.mean(err ** 2))),
        "per_joint": []
    }
    for j in range(desired.shape[1]):
        summary["per_joint"].append({
            "joint": j,
            "mean_abs_desired": float(np.mean(np.abs(desired[:, j]))),
            "mean_abs_actual":  float(np.mean(np.abs(actual[:, j]))),
            "mean_abs_error":   float(np.mean(np.abs(err[:, j]))),
            "rmse_error":       float(np.sqrt(np.mean(err[:, j] ** 2))),
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Per-joint hexbin scatter — Brian's standard figure
    nj = desired.shape[1]
    ncols = min(4, nj); nrows = int(np.ceil(nj / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4*nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for j in range(nj):
        ax = axes[j]
        ax.hexbin(desired[:, j], actual[:, j], gridsize=70, bins="log", mincnt=1)
        lo = float(min(np.min(desired[:, j]), np.min(actual[:, j])))
        hi = float(max(np.max(desired[:, j]), np.max(actual[:, j])))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"joint {j}"); ax.set_xlabel("desired Δq"); ax.set_ylabel("actual Δq")
    for j in range(nj, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{args.task}: NN→OSC one-step adapter accuracy")
    fig.savefig(out / "desired_vs_actual.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
