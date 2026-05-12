import collections
import json
import math
import os
import pathlib
import time

import dill
import h5py
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from reverse_controller.common import load_inverse_checkpoint

import robomimic.utils.file_utils as FileUtils


def _ordered_union(*key_lists):
    result = []
    for keys in key_lists:
        for key in keys:
            if key not in result:
                result.append(key)
    return result


class RobomimicJointOSCAdapterRunner(BaseLowdimRunner):
    """Run a joint-delta DP through a learned inverse adapter to OSC_POSE.

    The policy receives exactly the lowdim observation keys it was trained on.
    The adapter may receive a larger lowdim state, e.g. including joint velocity.
    The environment is instantiated with the union of both key sets and this
    runner slices the flat observation for the policy and adapter separately.
    """

    def __init__(
            self,
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256, 256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            adapter_checkpoint=None,
            adapter_obs_keys=None,
            adapter_device='cpu',
            use_policy_gripper=True,
            command_scale=1.0,
            **kwargs):
        super().__init__(output_dir)
        if adapter_checkpoint is None:
            raise ValueError("adapter_checkpoint is required.")
        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = os.path.expanduser(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        adapter_payload, adapter_model, adapter_normalizer = load_inverse_checkpoint(
            adapter_checkpoint,
            device=adapter_device)
        adapter_metadata = adapter_payload.get("dataset_metadata", {})
        if adapter_obs_keys is None:
            adapter_obs_keys = adapter_metadata.get("obs_keys", list(obs_keys))

        self.policy_obs_keys = list(obs_keys)
        self.adapter_obs_keys = list(adapter_obs_keys)
        self.env_obs_keys = _ordered_union(self.policy_obs_keys, self.adapter_obs_keys)

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, obs_keys=self.env_obs_keys)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=self.env_obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1),
                    file_path=None,
                    steps_per_render=max(20 // fps, 1)),
                n_obs_steps=n_obs_steps + n_latency_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps)

        env_fns = [env_fn] * n_envs
        env_seeds = []
        env_prefixs = []
        env_init_fn_dills = []
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media',
                            wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=True, exist_ok=True)
                        env.env.file_path = str(filename)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media',
                        wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        obs_slices = {}
        offset = 0
        with h5py.File(dataset_path, 'r') as f:
            first_obs = f['data/demo_0/obs']
            for key in self.env_obs_keys:
                dim = int(first_obs[key].shape[-1])
                obs_slices[key] = slice(offset, offset + dim)
                offset += dim

        self.env_meta = env_meta
        self.env = AsyncVectorEnv(env_fns)
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_slices = obs_slices
        self.policy_slices = [obs_slices[key] for key in self.policy_obs_keys]
        self.adapter_slices = [obs_slices[key] for key in self.adapter_obs_keys]
        self.adapter_checkpoint = adapter_checkpoint
        self.adapter_model = adapter_model
        self.adapter_normalizer = adapter_normalizer
        self.command_scale = float(command_scale)
        self.use_policy_gripper = bool(use_policy_gripper)

    def _slice_obs(self, obs, slices):
        return np.concatenate([obs[..., sl] for sl in slices], axis=-1).astype(np.float32)

    def _merge_policy_gripper(self, osc_command, policy_action):
        if not self.use_policy_gripper:
            return osc_command
        # Single-arm layout: joint-delta policy [dq(7), gripper], OSC [dpos(3), drot(3), gripper].
        if osc_command.shape[-1] == 7 and policy_action.shape[-1] >= 8:
            osc_command[..., 6] = policy_action[..., 7]
        return osc_command

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        adapter = self.adapter_model.to(device)
        normalizer = {
            name: {key: value.to(device) for key, value in stats.items()}
            for name, stats in self.adapter_normalizer.items()
        }
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        action_dim = getattr(adapter, "output_dim", None)
        if action_dim is None:
            for module in adapter.modules():
                if isinstance(module, torch.nn.Linear):
                    action_dim = module.out_features

        progress_path = pathlib.Path(self.output_dir) / "eval_progress.jsonl"
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        def log_progress(**row):
            row = {
                "time": time.time(),
                **row,
            }
            with open(progress_path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print("[eval_progress] " + " ".join(
                f"{key}={value}" for key, value in row.items() if key != "time"),
                flush=True)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            if len(this_init_fns) < n_envs:
                this_init_fns.extend([self.env_init_fn_dills[0]] * (n_envs - len(this_init_fns)))

            log_progress(
                event="chunk_start",
                chunk=chunk_idx + 1,
                n_chunks=n_chunks,
                active_envs=this_n_active_envs)
            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {self.env_meta['env_name']}JointOSCAdapter {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec)

            done = False
            step_count = 0
            while not done:
                policy_obs = self._slice_obs(
                    obs[:, :self.n_obs_steps],
                    self.policy_slices)
                np_obs_dict = {'obs': policy_obs}
                if self.past_action and past_action is not None:
                    np_obs_dict['past_action'] = past_action[:, -(self.n_obs_steps - 1):].astype(np.float32)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())['action']
                action = action[:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action.")

                adapter_state = self._slice_obs(
                    obs[:, self.n_obs_steps - 1],
                    self.adapter_slices)
                batch_size, horizon, action_size = action.shape
                desired_delta = action[..., :7]
                state_tile = np.broadcast_to(
                    adapter_state[:, None, :],
                    (batch_size, horizon, adapter_state.shape[-1]))
                adapter_input = np.concatenate([state_tile, desired_delta], axis=-1)
                adapter_input = adapter_input.reshape(batch_size * horizon, -1)

                with torch.no_grad():
                    tx = torch.as_tensor(adapter_input, dtype=torch.float32, device=device)
                    tx = (tx - normalizer['input']['mean']) / normalizer['input']['std']
                    pred = adapter(tx)
                    pred = pred * normalizer['command']['std'] + normalizer['command']['mean']
                osc_action = pred.detach().cpu().numpy().reshape(batch_size, horizon, action_dim)
                osc_action = np.clip(osc_action, -self.command_scale, self.command_scale).astype(np.float32)
                osc_action = self._merge_policy_gripper(osc_action, action)

                obs, reward, done, info = env.step(osc_action)
                done = np.all(done)
                past_action = action
                step_count += action.shape[1]
                pbar.update(action.shape[1])
                if step_count % 40 == 0 or done:
                    log_progress(
                        event="chunk_step",
                        chunk=chunk_idx + 1,
                        n_chunks=n_chunks,
                        step=step_count,
                        max_steps=self.max_steps,
                        active_envs=this_n_active_envs,
                        done_envs=int(np.sum(done)))
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            log_progress(event="chunk_done", chunk=chunk_idx + 1, n_chunks=n_chunks)

        max_rewards = collections.defaultdict(list)
        log_data = {}
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f'sim_video_{seed}'] = wandb.Video(video_path)
        for prefix, value in max_rewards.items():
            log_data[prefix + 'mean_score'] = float(np.mean(value))
        log_progress(event="run_done")
        return log_data
