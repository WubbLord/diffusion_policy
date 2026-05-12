"""Runner that uses Brian's InverseControllerMLP as the adapter from
joint-delta DP output to OSC_POSE controller action.

Pipeline per step:
    state         = obs concatenated by obs_keys at current timestep
    Δq_target     = policy.predict_action(obs)['action'][..., :7n] (per arm)
    OSC command   = NN(concat(state, Δq_target))
                  (in OSC's normalized [-1, 1] space; gripper included)
    env.step(OSC command)
"""
import os, copy, math, pathlib, dill, h5py, wandb, collections, tqdm
import numpy as np
import torch
import wandb.sdk.data_types.video as wv

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils

from reverse_controller.common import (
    load_inverse_checkpoint, predict_command, build_state_features,
)


class RobomimicJointBrianOSCRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir, dataset_path, obs_keys,
            n_train=10, n_train_vis=3, train_start_idx=0,
            n_test=22, n_test_vis=6, test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2, n_action_steps=8, n_latency_steps=0,
            render_hw=(256, 256), render_camera_name='agentview',
            fps=10, crf=22, past_action=False,
            tqdm_interval_sec=5.0, n_envs=None,
            # Brian-style adapter:
            adapter_path: str = None,
            adapter_obs_keys: list = None,   # the obs_keys used to build state features at training time
            command_scale: float = 1.0,       # OSC actions are normalized; scale=1
        ):
        super().__init__(output_dir)
        if n_envs is None: n_envs = n_train + n_test
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps
        dataset_path = os.path.expanduser(dataset_path)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicLowdimWrapper(
                        env=robomimic_env, obs_keys=obs_keys, init_state=None,
                        render_hw=render_hw, render_camera_name=render_camera_name),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264', input_pix_fmt='rgb24',
                        crf=crf, thread_type='FRAME', thread_count=1),
                    file_path=None,
                    steps_per_render=max(20 // fps, 1)),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps)

        env_fns = [env_fn] * n_envs
        env_seeds, env_prefixs, env_init_fn_dills = [], [], []
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]
                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    env.env.video_recoder.stop(); env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath('media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    env.env.env.init_state = init_state
                env_seeds.append(train_idx); env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis
            def init_fn(env, seed=seed, enable_render=enable_render):
                env.env.video_recoder.stop(); env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath('media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                env.env.env.init_state = None
                env.seed(seed)
            env_seeds.append(seed); env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # Compute obs slices for state features
        with h5py.File(dataset_path, 'r') as f:
            first_obs = f['data/demo_0/obs']
            obs_slices = {}
            offset = 0
            for k in obs_keys:
                d = int(first_obs[k].shape[-1])
                obs_slices[k] = slice(offset, offset + d)
                offset += d

        # Load Brian's InverseControllerMLP checkpoint
        assert adapter_path is not None, "adapter_path required"
        adapter_obs_keys = list(adapter_obs_keys) if adapter_obs_keys else list(obs_keys)
        payload, model, normalizer = load_inverse_checkpoint(adapter_path, device='cpu')
        # Move to GPU at run() time.

        self.env_meta = env_meta
        self.env = env; self.env_fns = env_fns
        self.env_seeds = env_seeds; self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps; self.crf = crf
        self.n_obs_steps = n_obs_steps; self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action; self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_slices = obs_slices
        self.adapter_obs_keys = adapter_obs_keys
        for k in adapter_obs_keys:
            if k not in obs_slices:
                raise KeyError(f"adapter_obs_keys requires {k!r} which is not in obs_keys")
        self.adapter_payload = payload
        self.adapter_model = model
        self.adapter_normalizer = normalizer
        self.command_scale = float(command_scale)

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        # Move adapter to GPU and move normalizer stats to GPU
        adapter = self.adapter_model.to(device)
        norm = {
            name: {k: v.to(device) for k, v in stats.items()}
            for name, stats in self.adapter_normalizer.items()
        }
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # State slice indices for adapter input
        state_sls = [self.obs_slices[k] for k in self.adapter_obs_keys]
        action_dim = adapter.output_dim if hasattr(adapter, 'output_dim') else None
        if action_dim is None:
            # introspect from last linear layer
            for m in adapter.modules():
                if isinstance(m, torch.nn.Linear):
                    action_dim = m.out_features

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval {self.env_meta['env_name']}JointBrianOSC {chunk_idx + 1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                np_obs_dict = {'obs': obs[:, :self.n_obs_steps].astype(np.float32)}
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[:, -(self.n_obs_steps - 1):].astype(np.float32)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                # Build state from current obs (at latest obs step)
                state_at_now = np.concatenate(
                    [obs[:, self.n_obs_steps - 1, sl] for sl in state_sls], axis=-1
                ).astype(np.float32)  # (B, state_dim)

                # For each step in the chunk, run NN to predict OSC command.
                # The DP action layout for joint-delta is [Δq(7) ... grip(1)] per arm.
                # For Brian's adapter, the "desired_delta" is the joint delta the
                # NN should invert. We feed the per-step Δq from the policy chunk.
                B, T, A_dim = action.shape
                osc_chunk = np.zeros((B, T, action_dim), dtype=np.float32)
                # We assume single-arm here (Δq is 7 dims, then 1 grip).
                # For dual-arm a separate runner / adapter would be needed.
                n_joint_dims = A_dim - 1
                dq = action[..., :n_joint_dims]                  # (B, T, 7)

                # Repeat state per timestep
                state_tile = np.broadcast_to(state_at_now[:, None, :], (B, T, state_at_now.shape[1]))
                # Build adapter input: concat(state, Δq) per (B,T)
                inp = np.concatenate([state_tile, dq], axis=-1).reshape(B * T, -1)
                with torch.no_grad():
                    tx = torch.as_tensor(inp, dtype=torch.float32, device=device)
                    tx_norm = (tx - norm['input']['mean']) / norm['input']['std']
                    pred = adapter(tx_norm)
                    pred = pred * norm['command']['std'] + norm['command']['mean']
                cmd = pred.detach().cpu().numpy().reshape(B, T, action_dim)
                # OSC actions live in [-1, 1] — clip for safety.
                cmd = np.clip(cmd, -1.0 * self.command_scale, 1.0 * self.command_scale).astype(np.float32)
                osc_chunk[...] = cmd

                obs, reward, done, info = env.step(osc_chunk)
                done = np.all(done)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

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
        return log_data
