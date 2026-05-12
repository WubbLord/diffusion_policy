"""B-2: Adapter-only oracle eval for the OSC NN adapter.

Replays held-out demo's recorded joint deltas through the learned NN adapter
to produce OSC commands, then steps the env. Skips the policy entirely.
Tests whether the adapter alone can execute known-good trajectories.

Usage:
    python oracle_replay_osc.py --task can_ph \\
        --adapter data/reverse_controller_osc/can_ph/inverse_mlp/best.pt \\
        --output-dir <out> --demo-start 100 --demo-end 150
"""
import argparse, json, pathlib, sys
import h5py
import numpy as np
import torch
import tqdm

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils

from reverse_controller.common import (
    load_inverse_checkpoint, DEFAULT_OBS_KEYS, build_state_features, parse_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. can_ph")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--demo-start", type=int, default=100)
    ap.add_argument("--demo-end", type=int, default=150)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--obs-keys", default=",".join(DEFAULT_OBS_KEYS))
    ap.add_argument("--joint-keys", default="robot0_joint_pos")
    args = ap.parse_args()

    dataset_path = f"data/robomimic/datasets/{args.task.replace('_', '/', 1)}/low_dim.hdf5"
    print(f"dataset = {dataset_path}")
    obs_keys = parse_csv(args.obs_keys)
    joint_keys = parse_csv(args.joint_keys)

    # Build env
    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': obs_keys})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
    print(f"env: {env_meta['env_name']}  action_dim={env.action_dimension}")

    # Load adapter
    device = torch.device(args.device)
    payload, model, normalizer = load_inverse_checkpoint(args.adapter, device=device)
    model.eval()
    print(f"adapter loaded; input_dim={payload['model_config']['input_dim']}  output_dim={payload['model_config']['output_dim']}")

    # Iterate held-out demos
    success_count = 0
    total = 0
    delta_mae_sum = 0.0
    n_steps_total = 0

    with h5py.File(dataset_path, "r") as f:
        all_demos = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[-1]))
        demo_names = all_demos[args.demo_start:args.demo_end]
        print(f"evaluating {len(demo_names)} demos: {demo_names[0]}..{demo_names[-1]}")

        for demo_name in tqdm.tqdm(demo_names, desc="oracle replay"):
            demo = f[f"data/{demo_name}"]
            init_state = np.asarray(demo["states"][0])
            obs_seq = {k: np.asarray(demo["obs"][k]) for k in obs_keys}
            actions_recorded = np.asarray(demo["actions"][:], dtype=np.float32)
            q_demo = np.concatenate(
                [np.asarray(demo["obs"][k][:], dtype=np.float32) for k in joint_keys],
                axis=-1)
            n_steps = actions_recorded.shape[0]

            env.reset()
            env.reset_to({"states": init_state})

            success = False
            for t in range(n_steps - 1):
                # Build state from current env obs
                env_obs = env.get_observation()
                state = np.concatenate(
                    [np.asarray(env_obs[k], dtype=np.float32).reshape(-1) for k in obs_keys],
                    axis=0)
                # Δq target from demo
                desired_dq = (q_demo[t + 1] - q_demo[t]).astype(np.float32)
                # Compute OSC command via adapter
                inp = np.concatenate([state, desired_dq], axis=-1)
                with torch.no_grad():
                    tx = torch.as_tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
                    tx_norm = (tx - normalizer['input']['mean']) / normalizer['input']['std']
                    pred = model(tx_norm)
                    cmd = pred * normalizer['command']['std'] + normalizer['command']['mean']
                cmd = cmd.detach().cpu().numpy().reshape(-1)
                # Clip OSC normalized action
                cmd = np.clip(cmd, -1.0, 1.0).astype(np.float32)
                # Step env
                env.step(cmd)
                # Track delta MAE
                q_curr = np.concatenate(
                    [np.asarray(env.get_observation()[k], dtype=np.float32).reshape(-1) for k in joint_keys],
                    axis=-1)
                # actual delta
                if t > 0:
                    pass  # for simplicity we just track per-step done
                # Success check
                try:
                    s = env.is_success()
                    if isinstance(s, dict): s = bool(s.get("task", any(s.values())))
                    else: s = bool(s)
                except Exception:
                    s = False
                if s:
                    success = True
                    break

            if success:
                success_count += 1
            total += 1

    summary = {
        "task": args.task,
        "adapter": args.adapter,
        "demo_range": [args.demo_start, args.demo_end],
        "n_demos": total,
        "n_success": success_count,
        "success_rate": success_count / max(total, 1),
    }
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle_replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
