import copy
import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from collections.abc import Sequence

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

import robomimic.utils.file_utils as FileUtils
from robosuite.controllers import load_controller_config


def _is_sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _expand_int_per_robot(value, n_robots, default, name):
    if value is None:
        return [default for _ in range(n_robots)]
    if not _is_sequence(value):
        return [int(value) for _ in range(n_robots)]
    result = [int(x) for x in value]
    if len(result) != n_robots:
        raise ValueError(
            f"{name} must have one entry per robot. "
            f"Expected {n_robots}, got {len(result)}.")
    return result


def _expand_scale_per_robot(value, joint_dims):
    n_robots = len(joint_dims)
    if not _is_sequence(value):
        return [
            np.full(joint_dim, float(value), dtype=np.float32)
            for joint_dim in joint_dims
        ]

    values = list(value)
    if len(values) == n_robots:
        result = list()
        for robot_idx, (robot_value, joint_dim) in enumerate(zip(values, joint_dims)):
            if _is_sequence(robot_value):
                scale = np.asarray(list(robot_value), dtype=np.float32)
                if scale.shape != (joint_dim,):
                    raise ValueError(
                        "joint_delta_scale nested entries must match each "
                        f"robot joint dimension. Robot {robot_idx} expected "
                        f"{joint_dim}, got {scale.shape}.")
            else:
                scale = np.full(joint_dim, float(robot_value), dtype=np.float32)
            result.append(scale)
        return result

    total_joint_dim = sum(joint_dims)
    if len(values) == total_joint_dim:
        values = np.asarray(values, dtype=np.float32)
        result = list()
        offset = 0
        for joint_dim in joint_dims:
            result.append(values[offset:offset + joint_dim])
            offset += joint_dim
        return result

    raise ValueError(
        "joint_delta_scale must be a scalar, one scalar/list per robot, "
        f"or one value per joint. Got {len(values)} values for "
        f"{n_robots} robots and {total_joint_dim} joints.")


def _make_joint_position_controller_configs(joint_delta_scales):
    controller_configs = list()
    for scale in joint_delta_scales:
        if np.any(scale <= 0):
            raise ValueError("joint_delta_scale values must be positive.")
        controller_config = load_controller_config(
            default_controller='JOINT_POSITION')
        controller_config = copy.deepcopy(controller_config)
        controller_config['output_max'] = scale.tolist()
        controller_config['output_min'] = (-scale).tolist()
        controller_configs.append(controller_config)
    if len(controller_configs) == 1:
        return controller_configs[0]
    return controller_configs


class RobomimicJointLowdimRunner(BaseLowdimRunner):
    """Robomimic low-dim runner for joint-delta policies.

    The policy predicts physical joint deltas followed by gripper commands:
    [robot0_dq, robot1_dq, ..., robot0_gripper, robot1_gripper, ...].
    Robosuite's JOINT_POSITION controller expects each robot action in
    normalized controller space:
    [robot0_controller_action, robot0_gripper, robot1_controller_action, ...].
    """

    def __init__(self,
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
            n_robots=None,
            joint_dims=None,
            gripper_dims=None,
            joint_delta_scale=0.05,
            joint_action_mode='delta',
            input_action_layout='joints_then_grippers',
            clip_joint_action=True,
            clip_gripper_action=True):
        super().__init__(output_dir)

        if joint_action_mode != 'delta':
            raise ValueError(
                "RobomimicJointLowdimRunner currently supports only "
                f"joint_action_mode='delta', got {joint_action_mode!r}.")
        if input_action_layout not in {'joints_then_grippers', 'interleaved'}:
            raise ValueError(
                "input_action_layout must be 'joints_then_grippers' or "
                f"'interleaved', got {input_action_layout!r}.")

        if n_envs is None:
            n_envs = n_train + n_test

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        env_meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(
            dataset_path))
        robot_names = env_meta['env_kwargs'].get('robots', [])
        if n_robots is None:
            n_robots = len(robot_names)
        if n_robots <= 0:
            raise ValueError("n_robots must be positive.")
        if len(robot_names) != n_robots:
            raise ValueError(
                f"Dataset env metadata has {len(robot_names)} robots, "
                f"but n_robots={n_robots}.")

        joint_dims = _expand_int_per_robot(
            joint_dims, n_robots=n_robots, default=7, name='joint_dims')
        gripper_dims = _expand_int_per_robot(
            gripper_dims, n_robots=n_robots, default=1, name='gripper_dims')
        joint_delta_scales = _expand_scale_per_robot(
            joint_delta_scale, joint_dims=joint_dims)

        env_meta['env_kwargs']['controller_configs'] = (
            _make_joint_position_controller_configs(joint_delta_scales))

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                obs_keys=obs_keys)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
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
                    steps_per_render=steps_per_render),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps)

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                        enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_robots = n_robots
        self.joint_dims = joint_dims
        self.gripper_dims = gripper_dims
        self.joint_delta_scales = joint_delta_scales
        self.input_action_layout = input_action_layout
        self.clip_joint_action = clip_joint_action
        self.clip_gripper_action = clip_gripper_action
        self.expected_action_dim = sum(joint_dims) + sum(gripper_dims)

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

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
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}JointLowdim {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                np_obs_dict = {
                    'obs': obs[:, :self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                env_action = self.transform_action(action)
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call(
                'get_attr', 'reward')[this_local_slice]

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def transform_action(self, action):
        if action.shape[-1] != self.expected_action_dim:
            raise RuntimeError(
                "Joint runner got invalid action dimension. Expected "
                f"{self.expected_action_dim}, got {action.shape[-1]}.")

        parts = list()
        if self.input_action_layout == 'joints_then_grippers':
            joint_offset = 0
            gripper_offset = sum(self.joint_dims)
            for robot_idx in range(self.n_robots):
                joint_dim = self.joint_dims[robot_idx]
                gripper_dim = self.gripper_dims[robot_idx]
                joint_delta = action[
                    ..., joint_offset:joint_offset + joint_dim]
                gripper = action[
                    ..., gripper_offset:gripper_offset + gripper_dim]
                parts.extend([
                    self._joint_delta_to_controller_action(
                        joint_delta, robot_idx),
                    self._format_gripper_action(gripper)])
                joint_offset += joint_dim
                gripper_offset += gripper_dim
        else:
            offset = 0
            for robot_idx in range(self.n_robots):
                joint_dim = self.joint_dims[robot_idx]
                gripper_dim = self.gripper_dims[robot_idx]
                joint_delta = action[..., offset:offset + joint_dim]
                offset += joint_dim
                gripper = action[..., offset:offset + gripper_dim]
                offset += gripper_dim
                parts.extend([
                    self._joint_delta_to_controller_action(
                        joint_delta, robot_idx),
                    self._format_gripper_action(gripper)])

        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _joint_delta_to_controller_action(self, joint_delta, robot_idx):
        scale = self.joint_delta_scales[robot_idx]
        controller_action = joint_delta / scale
        if self.clip_joint_action:
            controller_action = np.clip(controller_action, -1.0, 1.0)
        return controller_action

    def _format_gripper_action(self, gripper):
        if self.clip_gripper_action:
            gripper = np.clip(gripper, -1.0, 1.0)
        return gripper
