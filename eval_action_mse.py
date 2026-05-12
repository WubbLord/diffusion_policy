"""B-1: DP-only eval. Loads a checkpoint, runs the policy on the validation set,
reports per-joint and gripper action MSE compared to ground-truth Δq.

Usage:
    python eval_action_mse.py --checkpoint <path>.ckpt --output_json <path>.json
"""
import argparse, json, pathlib, sys
import numpy as np
import torch, dill, hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", type=int, default=200,
                   help="how many val samples to evaluate (each is a full action chunk)")
    p.add_argument("--use_ema", action="store_true", default=True)
    args = p.parse_args()

    print(f"loading {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # Build dataset and val split
    print("instantiating dataset (this loads the full hdf5)")
    ds = hydra.utils.instantiate(cfg.task.dataset)
    val_ds = ds.get_validation_dataset()
    n_val = len(val_ds)
    n = min(args.n_samples, n_val)
    print(f"val set has {n_val} samples; using {n}")

    # Build policy and load weights
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(ds.get_normalizer())
    sd = payload["state_dicts"]
    ema_sd = sd.get("ema_model")
    if ema_sd is not None and args.use_ema:
        policy.load_state_dict(ema_sd); used = "ema"
    else:
        policy.load_state_dict(sd["model"]); used = "model"
    print(f"loaded weights ({used})")

    device = torch.device(args.device)
    policy.to(device).eval()

    # Aggregate MSE over n samples in mini-batches
    n_act = int(cfg.n_action_steps)
    n_obs = int(cfg.n_obs_steps)

    sq_err_sum = None         # (action_dim,)
    abs_err_sum = None
    counts = 0
    BATCH = 16
    with torch.no_grad():
        for batch_start in range(0, n, BATCH):
            batch_end = min(n, batch_start + BATCH)
            samples = [val_ds[i] for i in range(batch_start, batch_end)]
            obs = torch.stack([s["obs"] for s in samples], dim=0).float().to(device)
            gt_action = torch.stack([s["action"] for s in samples], dim=0).float()  # (B, T, A)
            obs_for_pred = obs[:, :n_obs]
            out = policy.predict_action({"obs": obs_for_pred})
            pred_action = out["action"].detach().cpu()           # (B, n_act, A)
            gt_chunk = gt_action[:, n_obs - 1:n_obs - 1 + n_act] # align: action chunk starts at (n_obs-1)
            assert pred_action.shape == gt_chunk.shape, (pred_action.shape, gt_chunk.shape)
            err = (pred_action - gt_chunk).numpy()
            sq = (err ** 2).reshape(-1, err.shape[-1])           # (B*T, A)
            ab = np.abs(err).reshape(-1, err.shape[-1])
            if sq_err_sum is None:
                sq_err_sum = sq.sum(axis=0)
                abs_err_sum = ab.sum(axis=0)
            else:
                sq_err_sum += sq.sum(axis=0)
                abs_err_sum += ab.sum(axis=0)
            counts += sq.shape[0]
    mse = (sq_err_sum / counts).tolist()
    mae = (abs_err_sum / counts).tolist()

    # Label per-dim for joint-delta layout: action[..., :7] = Δq (joints),
    # action[..., 7] = gripper (single-arm). For dual-arm extend.
    action_dim = len(mse)
    n_joints = action_dim - 1
    labels = [f"joint{i}" for i in range(n_joints)] + ["gripper"]
    if action_dim == 16:  # dual-arm
        labels = [f"arm0_joint{i}" for i in range(7)] + [f"arm1_joint{i}" for i in range(7)] \
                 + ["arm0_gripper", "arm1_gripper"]

    out = {
        "checkpoint": args.checkpoint,
        "n_samples": int(counts),
        "n_action_steps": n_act,
        "action_dim": action_dim,
        "weights_used": used,
        "labels": labels,
        "mse_per_dim": mse,
        "mae_per_dim": mae,
        "mse_overall": float(np.mean(mse)),
        "mae_overall": float(np.mean(mae)),
        "mse_joints_only": float(np.mean(mse[:n_joints])),
        "mse_gripper_only": float(np.mean(mse[n_joints:])),
    }
    pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
