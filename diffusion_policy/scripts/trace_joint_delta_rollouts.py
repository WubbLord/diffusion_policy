import argparse
import copy
import json
import os
import pathlib
import sys
from collections import defaultdict, deque

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import dill
import gym
import hydra
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import tqdm

import robomimic.utils.file_utils as FileUtils

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
    RobomimicLowdimWrapper,
)
from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.env_runner.robomimic_joint_lowdim_runner import (
    _expand_scale_per_robot,
    _make_joint_position_controller_configs,
)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def _to_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _stack_last_n(items, n_steps):
    items = list(items)
    result = np.zeros((n_steps,) + items[-1].shape, dtype=items[-1].dtype)
    start = -min(n_steps, len(items))
    result[start:] = np.asarray(items[start:])
    if n_steps > len(items):
        result[:start] = result[start]
    return result


class SingleStepJointTraceWrapper(gym.Wrapper):
    """Single-action wrapper that returns stacked lowdim observations.

    This avoids MultiStepWrapper so we can record the actual joint transition
    for every individual action inside the policy action horizon.
    """

    def __init__(self, env, n_obs_steps, max_episode_steps, joint_key):
        super().__init__(env)
        self.n_obs_steps = n_obs_steps
        self.max_episode_steps = max_episode_steps
        self.joint_key = joint_key
        self.obs = deque(maxlen=n_obs_steps + 1)
        self.step_count = 0
        self.finished = False

        low = np.repeat(
            np.expand_dims(env.observation_space.low, axis=0),
            n_obs_steps,
            axis=0,
        )
        high = np.repeat(
            np.expand_dims(env.observation_space.high, axis=0),
            n_obs_steps,
            axis=0,
        )
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(n_obs_steps,) + env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = env.action_space

    def _joint_pos(self):
        raw_obs = self.env.env.get_observation()
        return raw_obs[self.joint_key].astype(np.float32).copy()

    def _get_obs(self):
        return _stack_last_n(self.obs, self.n_obs_steps)

    def reset(self):
        obs = self.env.reset()
        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        self.step_count = 0
        self.finished = False
        return self._get_obs()

    def step(self, action):
        if self.finished:
            info = {
                "q_before": self._joint_pos(),
                "q_after": self._joint_pos(),
                "actual_joint_delta": np.zeros_like(self._joint_pos()),
            }
            return self._get_obs(), 0.0, True, info

        q_before = self._joint_pos()
        obs, reward, done, info = self.env.step(action)
        q_after = self._joint_pos()

        self.step_count += 1
        if self.max_episode_steps is not None and self.step_count >= self.max_episode_steps:
            done = True
        self.finished = bool(done)

        self.obs.append(obs)
        info = dict(info)
        info["q_before"] = q_before
        info["q_after"] = q_after
        info["actual_joint_delta"] = q_after - q_before
        return self._get_obs(), reward, done, info

    def get_step_count(self):
        return self.step_count


def make_env_fn(env_meta, obs_keys, n_obs_steps, max_steps, joint_key):
    def env_fn():
        robomimic_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
        lowdim_env = RobomimicLowdimWrapper(
            env=robomimic_env,
            obs_keys=obs_keys,
            init_state=None,
            render_hw=(128, 128),
            render_camera_name="agentview",
        )
        return SingleStepJointTraceWrapper(
            lowdim_env,
            n_obs_steps=n_obs_steps,
            max_episode_steps=max_steps,
            joint_key=joint_key,
        )

    return env_fn


def transform_action(
    action,
    joint_dims,
    gripper_dims,
    joint_delta_scales,
    input_action_layout,
    clip_joint_action=True,
    clip_gripper_action=True,
):
    parts = []
    if input_action_layout == "joints_then_grippers":
        joint_offset = 0
        gripper_offset = sum(joint_dims)
        for robot_idx, (joint_dim, gripper_dim) in enumerate(zip(joint_dims, gripper_dims)):
            joint_delta = action[..., joint_offset:joint_offset + joint_dim]
            gripper = action[..., gripper_offset:gripper_offset + gripper_dim]
            controller_action = joint_delta / joint_delta_scales[robot_idx]
            if clip_joint_action:
                controller_action = np.clip(controller_action, -1.0, 1.0)
            if clip_gripper_action:
                gripper = np.clip(gripper, -1.0, 1.0)
            parts.extend([controller_action, gripper])
            joint_offset += joint_dim
            gripper_offset += gripper_dim
    elif input_action_layout == "interleaved":
        offset = 0
        for robot_idx, (joint_dim, gripper_dim) in enumerate(zip(joint_dims, gripper_dims)):
            joint_delta = action[..., offset:offset + joint_dim]
            offset += joint_dim
            gripper = action[..., offset:offset + gripper_dim]
            offset += gripper_dim
            controller_action = joint_delta / joint_delta_scales[robot_idx]
            if clip_joint_action:
                controller_action = np.clip(controller_action, -1.0, 1.0)
            if clip_gripper_action:
                gripper = np.clip(gripper, -1.0, 1.0)
            parts.extend([controller_action, gripper])
    else:
        raise ValueError(f"Unsupported input_action_layout={input_action_layout!r}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def append_record(records, key, value):
    records[key].append(np.asarray(value))


def records_to_arrays(records):
    arrays = {}
    for key, values in records.items():
        if not values:
            continue
        arrays[key] = np.concatenate(values, axis=0)
    return arrays


def summarize(arrays):
    pred = arrays["pred_joint_delta"]
    actual = arrays["actual_joint_delta"]
    eps = 1e-12

    summary = {
        "n_samples": int(pred.shape[0]),
        "joint_dim": int(pred.shape[1]),
        "success_rate": float(np.mean(arrays["rollout_max_reward"] > 0))
        if "rollout_max_reward" in arrays else None,
        "global": {},
        "per_joint": [],
    }

    x = pred.reshape(-1)
    y = actual.reshape(-1)
    slope0 = float(np.dot(x, y) / (np.dot(x, x) + eps))
    x_aug = np.stack([x, np.ones_like(x)], axis=-1)
    slope, intercept = np.linalg.lstsq(x_aug, y, rcond=None)[0]
    y_hat = x_aug @ np.array([slope, intercept])
    r2 = 1.0 - np.sum((y - y_hat) ** 2) / (np.sum((y - np.mean(y)) ** 2) + eps)
    summary["global"] = {
        "zero_intercept_slope": slope0,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "mean_abs_pred": float(np.mean(np.abs(x))),
        "mean_abs_actual": float(np.mean(np.abs(y))),
        "mean_abs_actual_over_pred": float(
            np.mean(np.abs(y)) / (np.mean(np.abs(x)) + eps)
        ),
    }

    for j in range(pred.shape[1]):
        xj = pred[:, j]
        yj = actual[:, j]
        slope0 = float(np.dot(xj, yj) / (np.dot(xj, xj) + eps))
        x_aug = np.stack([xj, np.ones_like(xj)], axis=-1)
        slope, intercept = np.linalg.lstsq(x_aug, yj, rcond=None)[0]
        corr = np.corrcoef(xj, yj)[0, 1] if np.std(xj) > 0 and np.std(yj) > 0 else np.nan
        y_hat = x_aug @ np.array([slope, intercept])
        r2 = 1.0 - np.sum((yj - y_hat) ** 2) / (np.sum((yj - np.mean(yj)) ** 2) + eps)
        summary["per_joint"].append({
            "joint": j,
            "zero_intercept_slope": slope0,
            "slope": float(slope),
            "intercept": float(intercept),
            "corr": float(corr),
            "r2": float(r2),
            "mean_abs_pred": float(np.mean(np.abs(xj))),
            "mean_abs_actual": float(np.mean(np.abs(yj))),
            "mean_abs_actual_over_pred": float(
                np.mean(np.abs(yj)) / (np.mean(np.abs(xj)) + eps)
            ),
        })

    x_aug = np.concatenate([pred, np.ones((pred.shape[0], 1), dtype=pred.dtype)], axis=1)
    coef = np.linalg.lstsq(x_aug, actual, rcond=None)[0]
    summary["linear_map_actual_from_pred"] = {
        "matrix": coef[:-1].T.tolist(),
        "bias": coef[-1].tolist(),
        "convention": "actual_joint_delta ~= matrix @ pred_joint_delta + bias",
    }
    return summary


def make_plots(arrays, summary, output_dir):
    pred = arrays["pred_joint_delta"]
    actual = arrays["actual_joint_delta"]
    joint_dim = pred.shape[1]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    for j in range(joint_dim):
        ax = axes[j]
        ax.hexbin(pred[:, j], actual[:, j], gridsize=80, bins="log", mincnt=1)
        lo = float(min(np.min(pred[:, j]), np.min(actual[:, j])))
        hi = float(max(np.max(pred[:, j]), np.max(actual[:, j])))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, label="actual=pred")
        sj = summary["per_joint"][j]
        ax.plot(
            [lo, hi],
            [sj["slope"] * lo + sj["intercept"], sj["slope"] * hi + sj["intercept"]],
            color="red",
            linewidth=1,
            label="linear fit",
        )
        ax.set_title(
            f"joint {j}: slope0={sj['zero_intercept_slope']:.3f}, "
            f"r={sj['corr']:.2f}"
        )
        ax.set_xlabel("predicted joint delta command target (rad)")
        ax.set_ylabel("actual joint delta after env.step (rad)")
    for j in range(joint_dim, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="best")
    fig.savefig(output_dir / "pred_vs_actual_joint_delta.png", dpi=180)
    plt.close(fig)

    slopes = [x["zero_intercept_slope"] for x in summary["per_joint"]]
    ratios = [x["mean_abs_actual_over_pred"] for x in summary["per_joint"]]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    width = 0.35
    idx = np.arange(joint_dim)
    ax.bar(idx - width / 2, slopes, width=width, label="zero-intercept slope")
    ax.bar(idx + width / 2, ratios, width=width, label="mean |actual| / |pred|")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xlabel("joint")
    ax.set_ylabel("actual / predicted")
    ax.set_title("Joint delta attenuation by joint")
    ax.legend(loc="best")
    fig.savefig(output_dir / "joint_delta_attenuation.png", dpi=180)
    plt.close(fig)


def save_outputs(records, rollout_max_rewards, output_dir, stem):
    arrays = records_to_arrays(records)
    if rollout_max_rewards:
        rollout_ids = np.asarray(sorted(rollout_max_rewards.keys()), dtype=np.int64)
        arrays["rollout_id"] = rollout_ids
        arrays["rollout_max_reward"] = np.asarray(
            [rollout_max_rewards[int(i)] for i in rollout_ids],
            dtype=np.float32,
        )
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
    if "pred_joint_delta" in arrays and arrays["pred_joint_delta"].shape[0] > 0:
        summary = summarize(arrays)
        with open(output_dir / f"{stem}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        if stem == "trace":
            make_plots(arrays, summary, output_dir)


def load_policy(checkpoint, output_dir, device):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace = workspace  # type: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    return cfg, policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rollouts", type=int, default=500)
    parser.add_argument("--n-envs", type=int, default=28)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--joint-key", default="robot0_joint_pos")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, output_dir, args.device)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps

    if cfg.get("past_action_visible", False):
        raise NotImplementedError("This trace script does not implement past_action conditioning.")

    runner_cfg = cfg.task.env_runner
    dataset_path = os.path.expanduser(str(runner_cfg.dataset_path))
    obs_keys = list(_to_container(runner_cfg.obs_keys))
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    max_steps = int(args.max_steps or runner_cfg.max_steps)
    start_seed = int(args.start_seed or runner_cfg.get("test_start_seed", 100000))

    joint_dims = [int(x) for x in _to_container(runner_cfg.get("joint_dims", [7]))]
    gripper_dims = [int(x) for x in _to_container(runner_cfg.get("gripper_dims", [1]))]
    joint_delta_scale = _to_container(runner_cfg.get("joint_delta_scale", 0.05))
    joint_delta_scales = _expand_scale_per_robot(joint_delta_scale, joint_dims=joint_dims)
    input_action_layout = str(runner_cfg.get("input_action_layout", "joints_then_grippers"))

    if len(joint_dims) != 1:
        raise NotImplementedError("This trace script currently handles one robot arm.")
    joint_dim = joint_dims[0]

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    env_meta["env_kwargs"]["controller_configs"] = _make_joint_position_controller_configs(
        joint_delta_scales
    )

    n_envs = min(args.n_envs, args.n_rollouts)
    env_fns = [
        make_env_fn(
            env_meta=env_meta,
            obs_keys=obs_keys,
            n_obs_steps=n_obs_steps,
            max_steps=max_steps,
            joint_key=args.joint_key,
        )
        for _ in range(n_envs)
    ]
    env = AsyncVectorEnv(env_fns)

    metadata = {
        "checkpoint": args.checkpoint,
        "dataset_path": dataset_path,
        "obs_keys": obs_keys,
        "n_rollouts": args.n_rollouts,
        "n_envs": n_envs,
        "start_seed": start_seed,
        "max_steps": max_steps,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "joint_dims": joint_dims,
        "gripper_dims": gripper_dims,
        "joint_delta_scales": [x.tolist() for x in joint_delta_scales],
        "input_action_layout": input_action_layout,
        "policy_num_inference_steps": int(policy.num_inference_steps),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    records = defaultdict(list)
    rollout_max_rewards = {}
    completed = 0

    try:
        for chunk_start in tqdm.trange(
            0,
            args.n_rollouts,
            n_envs,
            desc="trace rollout chunks",
        ):
            chunk_n = min(n_envs, args.n_rollouts - chunk_start)
            rollout_ids = np.arange(chunk_start, chunk_start + chunk_n, dtype=np.int64)
            seeds = [start_seed + int(i) for i in rollout_ids]
            if chunk_n < n_envs:
                seeds.extend([seeds[0]] * (n_envs - chunk_n))

            env.seed(seeds)
            obs = env.reset()
            policy.reset()

            active = np.zeros((n_envs,), dtype=bool)
            active[:chunk_n] = True
            step_counts = np.zeros((n_envs,), dtype=np.int64)
            max_rewards = np.full((n_envs,), -np.inf, dtype=np.float32)

            while np.any(active):
                obs_dict = {
                    "obs": torch.from_numpy(obs.astype(np.float32)).to(policy.device)
                }
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )["action"]

                for horizon_idx in range(n_action_steps):
                    if not np.any(active):
                        break

                    raw_action = action[:, horizon_idx]
                    env_action = transform_action(
                        raw_action,
                        joint_dims=joint_dims,
                        gripper_dims=gripper_dims,
                        joint_delta_scales=joint_delta_scales,
                        input_action_layout=input_action_layout,
                    )

                    was_active = active.copy()
                    obs, reward, done, infos = env.step(env_action)

                    for local_idx in np.flatnonzero(was_active[:chunk_n]):
                        info = infos[local_idx]
                        pred_joint_delta = raw_action[local_idx, :joint_dim]
                        controller_joint_action = env_action[local_idx, :joint_dim]
                        append_record(records, "rollout", [rollout_ids[local_idx]])
                        append_record(records, "seed", [seeds[local_idx]])
                        append_record(records, "step", [step_counts[local_idx]])
                        append_record(records, "horizon_index", [horizon_idx])
                        append_record(records, "pred_joint_delta", pred_joint_delta[None])
                        append_record(records, "controller_joint_action", controller_joint_action[None])
                        append_record(records, "actual_joint_delta", info["actual_joint_delta"][None])
                        append_record(records, "q_before", info["q_before"][None])
                        append_record(records, "q_after", info["q_after"][None])
                        append_record(records, "gripper_action", raw_action[local_idx, joint_dim:][None])
                        append_record(records, "reward", [reward[local_idx]])
                        append_record(records, "done", [done[local_idx]])

                    max_rewards[:chunk_n] = np.maximum(max_rewards[:chunk_n], reward[:chunk_n])
                    step_counts[was_active] += 1
                    active[:chunk_n] &= ~done[:chunk_n]
                    active[:chunk_n] &= step_counts[:chunk_n] < max_steps

            for local_idx, rollout_id in enumerate(rollout_ids):
                rollout_max_rewards[int(rollout_id)] = float(max_rewards[local_idx])
            completed += chunk_n
            save_outputs(records, rollout_max_rewards, output_dir, "trace_partial")
            print(f"completed {completed}/{args.n_rollouts} rollouts", flush=True)

    finally:
        env.close()

    save_outputs(records, rollout_max_rewards, output_dir, "trace")
    print(f"Wrote trace outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
