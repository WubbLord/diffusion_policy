"""Residual demo-supervised dataset for NN-OSC adapter on top of FK->OSC.

For each demo timestep t the saved label is the residual OSC command:

    fk_pred[t]   = FK->OSC(q[t], q[t+1]-q[t], grip[t])     # 7-D, normalized [-1,1]
    residual[t]  = demo_command[t] - fk_pred[t]            # 7-D training target

State and desired_delta are the same as the plain demo-only collector. At
inference the runner adds the NN(state, desired_delta) residual to the FK->OSC
command and clips back into [-1, 1].

Single-arm only. obs/joint keys are CLI args, default matches joint-delta DP obs.
"""
import argparse, copy, pathlib, os, sys
import h5py, numpy as np, tqdm
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import robomimic.utils.file_utils as FileUtils
from reverse_controller.common import (
    DEFAULT_OBS_KEYS, build_state_features, get_demo_names, parse_csv, save_json,
)
from diffusion_policy.env_runner.robomimic_joint_fk_to_eef_runner import _PandaFK


def _world_panda_R(panda_fk: _PandaFK, q0, quat0):
    _, R_fk0 = panda_fk.fk(np.asarray(q0, dtype=np.float64))
    R_e0 = Rotation.from_quat(np.asarray(quat0, dtype=np.float64)).as_matrix()
    return R_e0 @ R_fk0.T


def _fk_command_chunk(panda_fk: _PandaFK, q_curr, dq, gripper, R_world_panda,
                       delta_pos_clip=0.05, delta_rot_clip=0.5):
    """Compute one-step normalized OSC command via FK. q_curr (7,), dq (7,), gripper scalar."""
    p_prev, R_prev = panda_fk.fk(q_curr.astype(np.float64))
    q_next = q_curr.astype(np.float64) + dq.astype(np.float64)
    p_t, R_t = panda_fk.fk(q_next)
    dp_panda = p_t - p_prev
    dR_panda = R_t @ R_prev.T
    dr_panda = Rotation.from_matrix(dR_panda).as_rotvec()
    dp_world = R_world_panda @ dp_panda
    dr_world = R_world_panda @ dr_panda
    cmd = np.zeros(7, dtype=np.float32)
    cmd[0:3] = np.clip(dp_world / delta_pos_clip, -1.0, 1.0)
    cmd[3:6] = np.clip(dr_world / delta_rot_clip, -1.0, 1.0)
    cmd[6] = gripper
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--obs-keys", default=",".join(DEFAULT_OBS_KEYS))
    ap.add_argument("--joint-key", default="robot0_joint_pos")
    ap.add_argument("--eef-quat-key", default="robot0_eef_quat")
    ap.add_argument("--gripper-action-index", type=int, default=-1)
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_path = os.path.expanduser(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    obs_keys = parse_csv(args.obs_keys)
    joint_key = args.joint_key
    eef_quat_key = args.eef_quat_key

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    demo_names = get_demo_names(dataset_path)
    if args.max_demos is not None:
        demo_names = demo_names[:args.max_demos]

    panda_fk = _PandaFK()  # default panda XML

    metadata = {
        "dataset": dataset_path,
        "controller": "OSC_POSE",
        "supervision": "residual_demo_only",
        "obs_keys": obs_keys,
        "joint_key": joint_key,
        "joint_keys": [joint_key],
        "eef_quat_key": eef_quat_key,
        "n_robots": 1,
        "joint_delta_scale": [1.0] * 7,
        "samples_per_step": 1,
        "n_envs": 0,
        "seed": 0,
        "n_demos": len(demo_names),
        "demo_names": demo_names,
        "format": {
            "state": "build_state_features(obs, obs_keys, t)",
            "desired_delta": "q[t+1] - q[t]",
            "command": "demo_action[t] - FK->OSC(q[t], desired_delta, grip[t])  (residual)",
        },
        "fk_norm": {"delta_pos_clip": 0.05, "delta_rot_clip": 0.5},
    }
    save_json(output_dir / "metadata.json", metadata)

    n_total = 0
    for demo_idx, demo_name in enumerate(tqdm.tqdm(demo_names, desc="residual collect")):
        shard_path = shard_dir / f"{demo_name}.npz"
        if shard_path.exists() and not args.overwrite:
            n_total += 1
            continue

        with h5py.File(dataset_path, "r") as f:
            demo = f[f"data/{demo_name}"]
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
            obs = {k: np.asarray(demo["obs"][k][:], dtype=np.float32) for k in set(obs_keys) | {joint_key, eef_quat_key}}
            next_obs_q = np.asarray(demo["next_obs"][joint_key][:], dtype=np.float32)

        n_steps = actions.shape[0]
        q_t = obs[joint_key]                              # (n_steps, 7)
        q_tp1 = next_obs_q                                # (n_steps, 7)
        gripper_idx = args.gripper_action_index if args.gripper_action_index >= 0 else actions.shape[-1] + args.gripper_action_index

        # World<-panda calibration from the demo's first frame.
        R_world_panda = _world_panda_R(panda_fk, q_t[0], obs[eef_quat_key][0])

        state_features = np.stack(
            [build_state_features({k: obs[k] for k in obs}, obs_keys, t) for t in range(n_steps)], axis=0
        ).astype(np.float32)

        desired = (q_tp1 - q_t).astype(np.float32)        # (n_steps, 7)
        demo_grip = actions[:, gripper_idx].astype(np.float32)  # (n_steps,)

        # FK prediction per step (single arm, scalar gripper).
        fk_pred = np.zeros((n_steps, 7), dtype=np.float32)
        for t in range(n_steps):
            fk_pred[t] = _fk_command_chunk(
                panda_fk,
                q_t[t], desired[t], demo_grip[t],
                R_world_panda,
            )

        # Demo command vector in OSC layout. For single-arm tasks: action is 7-D
        # [Δp(3), Δr(3), grip(1)] directly.
        demo_cmd = actions[:, :7].astype(np.float32)
        residual = (demo_cmd - fk_pred).astype(np.float32)

        data = {
            "state": state_features,
            "desired_delta": desired,
            "command": residual,                         # training target = residual
            "demo_delta": desired.copy(),
            "fk_pred": fk_pred,                          # for debugging
            "demo_cmd": demo_cmd,                        # for debugging
            "demo_idx": np.full((n_steps,), demo_idx, dtype=np.int32),
            "timestep": np.arange(n_steps, dtype=np.int32),
            "sample_idx": np.zeros((n_steps,), dtype=np.int16),
        }
        tmp_path = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **data)
        tmp_path.replace(shard_path)
        n_total += 1

    save_json(output_dir / "DONE.json", {
        "complete": True, "n_shards": n_total,
        "controller": "OSC_POSE", "supervision": "residual_demo_only",
    })
    print(f"Wrote {n_total} residual shards to {shard_dir}")


if __name__ == "__main__":
    main()
