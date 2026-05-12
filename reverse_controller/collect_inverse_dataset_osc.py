"""OSC variant of Brian's collect_inverse_dataset.py.

Differences from JOINT_POSITION version:
- Uses env_meta's default OSC_POSE controller (no override).
- Commands are 7-dim OSC actions in OSC's normalized [-1, 1] space:
    [Δp_norm(3), Δr_norm(3), gripper(1)]
  OSC internally maps these to physical [-output_max, output_max] units.
- Probe applies the OSC command directly to env.step (no /scale division),
  reads joint position before and after to compute realized Δq.
- The dataset learned by train_inverse_model.py becomes:
    (state, desired_Δq) -> OSC command that produces that Δq
"""
import argparse
import copy
import os
import pathlib
import sys

import h5py
import gym
import numpy as np
import tqdm

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

import robomimic.utils.file_utils as FileUtils

from reverse_controller.common import (
    DEFAULT_OBS_KEYS,
    build_state_features,
    get_demo_names,
    parse_csv,
    parse_int_csv,
    save_json,
)


class JointProbeEnvOSC:
    """Same as Brian's JointProbeEnv but env runs OSC_POSE (env_meta default).

    Action layout matches what robosuite's OSC env expects: a flat array of
    [arm_control_dim + gripper_dim] per robot. For single-arm OSC_POSE this
    is 7 dims; for dual-arm it's 14 dims.
    """

    def __init__(self, env_meta, joint_keys):
        if isinstance(joint_keys, str):
            joint_keys = [joint_keys]
        self.joint_keys = list(joint_keys)
        self.env = create_env(env_meta=env_meta, obs_keys=self.joint_keys)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.env.action_dimension,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.metadata = {}

    def reset(self):
        self.env.reset()
        return np.zeros((1,), dtype=np.float32)

    def step(self, action):
        return self.reset(), 0.0, False, {}

    def close(self):
        close_fn = getattr(self.env, "close", None)
        if close_fn is not None:
            close_fn()

    def probe(self, sim_state, action, joint_keys=None):
        """Apply env_action at sim_state, return Δq across joint_keys."""
        keys = list(joint_keys) if joint_keys is not None else self.joint_keys
        self.env.reset_to({"states": sim_state})
        obs_before = self.env.get_observation()
        q_before = np.concatenate([
            obs_before[key].astype(np.float32).reshape(-1) for key in keys
        ], axis=0)
        self.env.step(action)
        obs_after = self.env.get_observation()
        q_after = np.concatenate([
            obs_after[key].astype(np.float32).reshape(-1) for key in keys
        ], axis=0)
        return q_after - q_before


def make_env_fn(env_meta, joint_keys):
    if isinstance(joint_keys, str):
        joint_keys = [joint_keys]

    def env_fn():
        return JointProbeEnvOSC(env_meta=env_meta, joint_keys=joint_keys)

    return env_fn


def sample_osc_commands(demo_action_at_t, n_samples, rng, n_robots=1):
    """Mirror Brian sample_commands but in OSC normalized [-1,1] space."""
    dim = demo_action_at_t.shape[-1]
    cmds = []
    # Brian-style anchors: large multipliers to span the bounded space.
    anchors = [
        np.zeros((dim,), dtype=np.float32),
        demo_action_at_t,
        2.0 * demo_action_at_t,
        4.0 * demo_action_at_t,
        8.0 * demo_action_at_t,
        16.0 * demo_action_at_t,
        -demo_action_at_t,
    ]
    for a in anchors:
        if len(cmds) >= n_samples: break
        cmds.append(a.astype(np.float32))
    while len(cmds) < n_samples:
        mode = rng.random()
        if mode < 0.35:
            cmd = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
        elif mode < 0.70:
            factor = rng.uniform(0.0, 20.0)
            noise = rng.normal(0.0, 0.10, size=dim).astype(np.float32)
            cmd = factor * demo_action_at_t + noise
        else:
            cmd = rng.normal(0.0, 0.35, size=dim).astype(np.float32)
        cmds.append(cmd.astype(np.float32))
    return np.clip(cmds, -1.0, 1.0).astype(np.float32)


def collect_demo(
    env, dataset_path, demo_name, demo_idx,
    obs_keys, joint_keys,
    samples_per_step, n_envs, rng, n_robots,
):
    with h5py.File(dataset_path, "r") as f:
        demo = f[f"data/{demo_name}"]
        states = np.asarray(demo["states"][:])
        actions = np.asarray(demo["actions"][:], dtype=np.float32)  # OSC commands recorded
        obs = demo["obs"]
        next_obs = demo["next_obs"]
        n_steps = actions.shape[0]

        state_features = []
        demo_delta = []
        commands = []
        env_actions = []
        demo_indices = []
        timestep_indices = []
        sample_indices = []

        for t in range(n_steps):
            state = build_state_features(obs, obs_keys, t)
            target_delta = np.concatenate([
                np.asarray(next_obs[joint_key][t], dtype=np.float32).reshape(-1)
                - np.asarray(obs[joint_key][t], dtype=np.float32).reshape(-1)
                for joint_key in joint_keys
            ], axis=0)
            demo_a = actions[t].astype(np.float32)  # full action (7 or 14 dims)
            sampled = sample_osc_commands(demo_a, samples_per_step, rng, n_robots=n_robots)
            for sample_idx, command in enumerate(sampled):
                state_features.append(state)
                demo_delta.append(target_delta)
                commands.append(command)
                env_actions.append(command)  # OSC actions go straight to env.step
                demo_indices.append(demo_idx)
                timestep_indices.append(t)
                sample_indices.append(sample_idx)

        state_features = np.asarray(state_features, dtype=np.float32)
        demo_delta = np.asarray(demo_delta, dtype=np.float32)
        commands = np.asarray(commands, dtype=np.float32)
        env_actions = np.asarray(env_actions, dtype=np.float32)
        demo_indices = np.asarray(demo_indices, dtype=np.int32)
        timestep_indices = np.asarray(timestep_indices, dtype=np.int32)
        sample_indices = np.asarray(sample_indices, dtype=np.int16)

        sim_states = np.repeat(states, samples_per_step, axis=0)

    actual_delta = np.zeros((commands.shape[0], demo_delta.shape[-1]), dtype=np.float32)
    for start in range(0, len(commands), n_envs):
        end = min(start + n_envs, len(commands))
        args_list = [
            (sim_states[i], env_actions[i], joint_keys)
            for i in range(start, end)
        ]
        if len(args_list) < n_envs:
            args_list.extend([args_list[0]] * (n_envs - len(args_list)))
        result = env.call_each("probe", args_list=args_list)
        actual_delta[start:end] = np.asarray(result[:end - start], dtype=np.float32)

    return {
        "state": state_features,
        "desired_delta": actual_delta,  # achieved Δq (what the NN predicts inverting)
        "command": commands,             # OSC normalized action that achieved it
        "demo_delta": demo_delta,
        "demo_idx": demo_indices,
        "timestep": timestep_indices,
        "sample_idx": sample_indices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obs-keys", default=",".join(DEFAULT_OBS_KEYS))
    parser.add_argument("--joint-key", default="robot0_joint_pos")
    parser.add_argument("--joint-keys", default=None)
    parser.add_argument("--samples-per-step", type=int, default=32)
    parser.add_argument("--n-envs", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_path = os.path.expanduser(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    obs_keys = parse_csv(args.obs_keys)
    joint_keys = parse_csv(args.joint_keys) if args.joint_keys is not None else [args.joint_key]
    n_robots = len(joint_keys)

    env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    # NB: do NOT override controller_configs — keep env_meta's default OSC_POSE.

    env_fn = make_env_fn(env_meta, joint_keys)
    env = AsyncVectorEnv([env_fn] * args.n_envs)

    demo_names = get_demo_names(dataset_path)
    if args.max_demos is not None:
        demo_names = demo_names[:args.max_demos]

    metadata = {
        "dataset": dataset_path,
        "controller": "OSC_POSE",
        "joint_delta_scale": [1.0] * (7 * n_robots),  # OSC action is already in [-1, 1]
        "obs_keys": obs_keys,
        "joint_keys": joint_keys,
        "n_robots": n_robots,
        "samples_per_step": args.samples_per_step,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "n_demos": len(demo_names),
        "demo_names": demo_names,
        "format": {
            "state": "full state features from obs_keys at timestep t",
            "desired_delta": "actual joint delta produced by probe (target for inverse map)",
            "command": "OSC_POSE normalized action [-1,1] (the inverse target)",
            "demo_delta": "logged next_obs[joint] - obs[joint]",
        },
    }
    save_json(output_dir / "metadata.json", metadata)

    rng = np.random.default_rng(args.seed)
    try:
        for demo_idx, demo_name in enumerate(tqdm.tqdm(demo_names, desc="collect demos")):
            shard_path = shard_dir / f"{demo_name}.npz"
            if shard_path.exists() and not args.overwrite:
                continue
            data = collect_demo(
                env=env,
                dataset_path=dataset_path,
                demo_name=demo_name,
                demo_idx=demo_idx,
                obs_keys=obs_keys,
                joint_keys=joint_keys,
                samples_per_step=args.samples_per_step,
                n_envs=args.n_envs,
                rng=rng,
                n_robots=n_robots,
            )
            tmp_path = shard_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, **data)
            tmp_path.replace(shard_path)
    finally:
        env.close()

    done = {
        "complete": True,
        "n_shards": len(list(shard_dir.glob("demo_*.npz"))),
        "controller": "OSC_POSE",
    }
    save_json(output_dir / "DONE.json", done)


if __name__ == "__main__":
    main()
