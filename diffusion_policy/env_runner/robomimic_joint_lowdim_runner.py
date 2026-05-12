import copy
import json
import os
import time
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
from reverse_controller.common import load_inverse_checkpoint, predict_command

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


class RobomimicJointAdapterLowdimWrapper(RobomimicLowdimWrapper):
    """Lowdim wrapper that turns desired joint deltas into JOINT_POSITION commands.

    The wrapped policy action is one desired joint-delta action plus gripper
    commands. At every low-level env.step, the wrapper reads the current raw
    robosuite observation, evaluates f(state, desired_dq), converts the
    physical command to normalized JOINT_POSITION controller space, interleaves
    robot grippers as robosuite expects, and steps the underlying env.
    """

    def __init__(
            self,
            env,
            obs_keys,
            adapter_checkpoint,
            adapter_obs_keys=None,
            adapter_device='cpu',
            joint_key='robot0_joint_pos',
            joint_dim=7,
            joint_dims=None,
            gripper_dim=1,
            gripper_dims=None,
            input_action_layout='joints_then_grippers',
            command_scale=None,
            adapter_execution_mode='one_step',
            adapter_inner_steps=1,
            init_state=None,
            render_hw=(256, 256),
            render_camera_name='agentview'):
        super().__init__(
            env=env,
            obs_keys=obs_keys,
            init_state=init_state,
            render_hw=render_hw,
            render_camera_name=render_camera_name)

        if input_action_layout not in {'joints_then_grippers', 'interleaved'}:
            raise ValueError(
                "input_action_layout must be 'joints_then_grippers' or "
                f"'interleaved', got {input_action_layout!r}.")
        if adapter_execution_mode not in {'one_step', 'closed_loop'}:
            raise ValueError(
                "adapter_execution_mode must be 'one_step' or 'closed_loop', "
                f"got {adapter_execution_mode!r}.")
        if int(adapter_inner_steps) < 1:
            raise ValueError("adapter_inner_steps must be at least 1.")

        payload, model, normalizer = load_inverse_checkpoint(
            adapter_checkpoint, device=adapter_device)
        metadata = payload.get('dataset_metadata', {})
        if joint_dims is None:
            joint_dims = metadata.get('joint_dims', [joint_dim])
        if gripper_dims is None:
            gripper_dims = [gripper_dim for _ in joint_dims]

        self.joint_dims = [int(x) for x in joint_dims]
        self.gripper_dims = [int(x) for x in gripper_dims]
        if len(self.gripper_dims) != len(self.joint_dims):
            raise ValueError(
                "gripper_dims must have one entry per robot. "
                f"Got {len(self.gripper_dims)} gripper dims and "
                f"{len(self.joint_dims)} joint dims.")
        self.joint_dim = int(sum(self.joint_dims))
        self.gripper_dim = int(sum(self.gripper_dims))
        self.n_robots = len(self.joint_dims)

        if adapter_obs_keys is None:
            adapter_obs_keys = metadata.get('obs_keys', [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
                'robot0_joint_pos',
                'robot0_joint_vel',
            ])
        if command_scale is None:
            command_scale = metadata.get(
                'joint_delta_scale', [0.25] * self.joint_dim)

        self.adapter_checkpoint = adapter_checkpoint
        self.adapter_obs_keys = list(adapter_obs_keys)
        self.adapter_model = model
        self.adapter_normalizer = normalizer
        self.joint_key = joint_key
        self.joint_keys = [
            f'robot{i}_joint_pos' for i in range(self.n_robots)
        ]
        if self.n_robots == 1:
            self.joint_keys[0] = joint_key
        missing_joint_keys = [
            key for key in self.joint_keys if key not in self.adapter_obs_keys
        ]
        if missing_joint_keys and adapter_execution_mode == 'closed_loop':
            raise ValueError(
                "Closed-loop adapter execution needs joint position keys in "
                f"adapter_obs_keys. Missing: {missing_joint_keys}")
        self.input_action_layout = input_action_layout
        self.adapter_execution_mode = adapter_execution_mode
        self.adapter_inner_steps = int(adapter_inner_steps)
        self.command_scale = np.asarray(command_scale, dtype=np.float32)
        if self.command_scale.shape == ():
            self.command_scale = np.full(
                (self.joint_dim,), float(self.command_scale), dtype=np.float32)
        if self.command_scale.shape != (self.joint_dim,):
            raise ValueError(
                f"command_scale must have shape ({self.joint_dim},), "
                f"got {self.command_scale.shape}.")
        self.command_scales = list()
        offset = 0
        for joint_dim in self.joint_dims:
            self.command_scales.append(
                self.command_scale[offset:offset + joint_dim])
            offset += joint_dim

    def _build_adapter_state(self, raw_obs):
        missing = [key for key in self.adapter_obs_keys if key not in raw_obs]
        if missing:
            raise KeyError(
                f"Current robomimic observation is missing adapter keys: {missing}")
        return np.concatenate([
            np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
            for key in self.adapter_obs_keys
        ], axis=0)

    def _current_joint_pos(self, raw_obs):
        missing = [key for key in self.joint_keys if key not in raw_obs]
        if missing:
            raise KeyError(
                f"Current robomimic observation is missing joint keys: {missing}")
        qpos = np.concatenate([
            np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
            for key in self.joint_keys
        ], axis=0)
        if qpos.shape != (self.joint_dim,):
            raise RuntimeError(
                f"Expected current joint position shape ({self.joint_dim},), "
                f"got {qpos.shape}.")
        return qpos

    def _adapter_delta_to_controller_action(
            self, desired_delta, grippers, raw_obs):
        state = self._build_adapter_state(raw_obs)
        command = predict_command(
            model=self.adapter_model,
            normalizer=self.adapter_normalizer,
            state=state[None],
            desired_delta=desired_delta[None],
            command_scale=self.command_scale,
        )[0].astype(np.float32)
        return self._format_controller_action(command, grippers)

    def _desired_action_to_controller_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        expected_dim = self.joint_dim + self.gripper_dim
        if action.shape[-1] != expected_dim:
            raise RuntimeError(
                "Adapter wrapper got invalid action dimension. Expected "
                f"{expected_dim}, got {action.shape[-1]}.")

        desired_delta, grippers = self._split_policy_action(action)
        raw_obs = self.env.get_observation()
        return self._adapter_delta_to_controller_action(
            desired_delta=desired_delta,
            grippers=grippers,
            raw_obs=raw_obs)

    def _split_policy_action(self, action):
        desired_parts = list()
        gripper_parts = list()

        if self.input_action_layout == 'joints_then_grippers':
            joint_offset = 0
            gripper_offset = self.joint_dim
            for joint_dim, gripper_dim in zip(self.joint_dims, self.gripper_dims):
                desired_parts.append(
                    action[joint_offset:joint_offset + joint_dim])
                gripper_parts.append(
                    action[gripper_offset:gripper_offset + gripper_dim])
                joint_offset += joint_dim
                gripper_offset += gripper_dim
        else:
            offset = 0
            for joint_dim, gripper_dim in zip(self.joint_dims, self.gripper_dims):
                desired_parts.append(action[offset:offset + joint_dim])
                offset += joint_dim
                gripper_parts.append(action[offset:offset + gripper_dim])
                offset += gripper_dim

        desired_delta = np.concatenate(desired_parts, axis=0).astype(np.float32)
        grippers = [
            np.asarray(x, dtype=np.float32)
            for x in gripper_parts
        ]
        return desired_delta, grippers

    def _format_controller_action(self, command, grippers):
        parts = list()
        offset = 0
        for joint_dim, scale, gripper in zip(
                self.joint_dims, self.command_scales, grippers):
            robot_command = command[offset:offset + joint_dim]
            offset += joint_dim
            arm_action = np.clip(robot_command / scale, -1.0, 1.0)
            gripper_action = np.clip(gripper, -1.0, 1.0)
            parts.extend([arm_action, gripper_action])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def step(self, action):
        if self.adapter_execution_mode == 'one_step':
            controller_action = self._desired_action_to_controller_action(action)
            raw_obs, reward, done, info = self.env.step(controller_action)
            obs = np.concatenate([
                raw_obs[key] for key in self.obs_keys
            ], axis=0)
            return obs, reward, done, info

        action = np.asarray(action, dtype=np.float32)
        expected_dim = self.joint_dim + self.gripper_dim
        if action.shape[-1] != expected_dim:
            raise RuntimeError(
                "Adapter wrapper got invalid action dimension. Expected "
                f"{expected_dim}, got {action.shape[-1]}.")

        desired_delta, grippers = self._split_policy_action(action)
        raw_obs = self.env.get_observation()
        q_target = self._current_joint_pos(raw_obs) + desired_delta

        rewards = list()
        done = False
        info = {}
        for _ in range(self.adapter_inner_steps):
            current_q = self._current_joint_pos(raw_obs)
            residual = (q_target - current_q).astype(np.float32)
            controller_action = self._adapter_delta_to_controller_action(
                desired_delta=residual,
                grippers=grippers,
                raw_obs=raw_obs)
            raw_obs, reward, done, info = self.env.step(controller_action)
            rewards.append(reward)
            if done:
                break

        reward = np.max(rewards) if rewards else 0.0
        info = dict(info)
        info['adapter_execution_mode'] = self.adapter_execution_mode
        info['adapter_inner_steps'] = len(rewards)
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info


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
            clip_gripper_action=True,
            adapter_checkpoint=None,
            adapter_obs_keys=None,
            adapter_device='cpu',
            adapter_joint_key='robot0_joint_pos',
            adapter_execution_mode='one_step',
            adapter_inner_steps=1):
        super().__init__(output_dir)

        if joint_action_mode != 'delta':
            raise ValueError(
                "RobomimicJointLowdimRunner currently supports only "
                f"joint_action_mode='delta', got {joint_action_mode!r}.")
        if input_action_layout not in {'joints_then_grippers', 'interleaved'}:
            raise ValueError(
                "input_action_layout must be 'joints_then_grippers' or "
                f"'interleaved', got {input_action_layout!r}.")
        if adapter_execution_mode not in {'one_step', 'closed_loop'}:
            raise ValueError(
                "adapter_execution_mode must be 'one_step' or 'closed_loop', "
                f"got {adapter_execution_mode!r}.")
        if int(adapter_inner_steps) < 1:
            raise ValueError("adapter_inner_steps must be at least 1.")

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
        env_obs_keys = list(obs_keys)
        if adapter_checkpoint is not None:
            payload, _, _ = load_inverse_checkpoint(
                adapter_checkpoint, device=adapter_device)
            adapter_metadata = payload.get('dataset_metadata', {})
            if adapter_obs_keys is None:
                adapter_obs_keys = adapter_metadata.get('obs_keys', [
                    'object',
                    'robot0_eef_pos',
                    'robot0_eef_quat',
                    'robot0_gripper_qpos',
                    'robot0_joint_pos',
                    'robot0_joint_vel',
                ])
            env_obs_keys = list(dict.fromkeys(
                list(obs_keys) + list(adapter_obs_keys)))
        else:
            adapter_metadata = {}

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                obs_keys=env_obs_keys)
            if adapter_checkpoint is not None:
                lowdim_env = RobomimicJointAdapterLowdimWrapper(
                    env=robomimic_env,
                    obs_keys=obs_keys,
                    adapter_checkpoint=adapter_checkpoint,
                    adapter_obs_keys=adapter_obs_keys,
                    adapter_device=adapter_device,
                    joint_key=adapter_joint_key,
                    joint_dim=sum(joint_dims),
                    joint_dims=joint_dims,
                    gripper_dim=sum(gripper_dims),
                    gripper_dims=gripper_dims,
                    input_action_layout=input_action_layout,
                    command_scale=np.concatenate(joint_delta_scales, axis=0),
                    adapter_execution_mode=adapter_execution_mode,
                    adapter_inner_steps=adapter_inner_steps,
                    init_state=None,
                    render_hw=render_hw,
                    render_camera_name=render_camera_name)
            else:
                lowdim_env = RobomimicLowdimWrapper(
                    env=robomimic_env,
                    obs_keys=obs_keys,
                    init_state=None,
                    render_hw=render_hw,
                    render_camera_name=render_camera_name)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    lowdim_env,
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
        self.adapter_checkpoint = adapter_checkpoint
        self.adapter_obs_keys = None if adapter_obs_keys is None else list(adapter_obs_keys)
        self.adapter_device = adapter_device
        self.adapter_metadata = adapter_metadata
        self.adapter_execution_mode = adapter_execution_mode
        self.adapter_inner_steps = int(adapter_inner_steps)

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        env_name = self.env_meta['env_name']
        progress_path = pathlib.Path(self.output_dir).joinpath("eval_progress.jsonl")

        def log_progress(event, **kwargs):
            record = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                "env_name": env_name,
                **kwargs,
            }
            message_items = [
                f"{key}={value}" for key, value in record.items()
                if key != "time"
            ]
            print("[eval_progress] " + " ".join(message_items), flush=True)
            try:
                with open(progress_path, "a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
            except Exception as exc:
                print(
                    f"[eval_progress] failed_to_write path={progress_path} error={exc}",
                    flush=True)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        log_progress(
            "run_start",
            n_envs=int(n_envs),
            n_inits=int(n_inits),
            n_chunks=int(n_chunks),
            max_steps=int(self.max_steps),
            n_action_steps=int(self.n_action_steps),
            adapter_execution_mode=str(self.adapter_execution_mode),
            adapter_inner_steps=int(self.adapter_inner_steps))

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

            chunk_start_time = time.monotonic()
            log_progress(
                "chunk_start",
                chunk=int(chunk_idx + 1),
                n_chunks=int(n_chunks),
                init_start=int(start),
                init_end=int(end),
                active_envs=int(this_n_active_envs))
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            log_progress(
                "chunk_reset_start",
                chunk=int(chunk_idx + 1),
                n_chunks=int(n_chunks))
            obs = env.reset()
            log_progress(
                "chunk_reset_done",
                chunk=int(chunk_idx + 1),
                n_chunks=int(n_chunks))
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}JointLowdim {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec)

            done = False
            last_log_step = 0
            last_log_time = time.monotonic()
            log_every_steps = max(1, 5 * self.n_action_steps)
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
                obs, reward, done_vec, info = env.step(env_action)
                done_arr = np.asarray(done_vec).reshape(-1)
                done = bool(np.all(done_arr))
                past_action = action

                pbar.update(action.shape[1])
                now = time.monotonic()
                if (
                        done
                        or (pbar.n - last_log_step) >= log_every_steps
                        or (now - last_log_time) >= 60.0):
                    reward_arr = np.asarray(reward).reshape(-1)
                    reward_arr = reward_arr[:this_n_active_envs]
                    log_progress(
                        "chunk_step",
                        chunk=int(chunk_idx + 1),
                        n_chunks=int(n_chunks),
                        step=int(min(pbar.n, self.max_steps)),
                        max_steps=int(self.max_steps),
                        active_envs=int(this_n_active_envs),
                        done_envs=int(np.sum(done_arr[:this_n_active_envs])),
                        reward_mean=float(np.mean(reward_arr)) if reward_arr.size else None,
                        elapsed_sec=float(now - chunk_start_time))
                    last_log_step = pbar.n
                    last_log_time = now
            pbar.close()

            log_progress(
                "chunk_render_start",
                chunk=int(chunk_idx + 1),
                n_chunks=int(n_chunks),
                elapsed_sec=float(time.monotonic() - chunk_start_time))
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call(
                'get_attr', 'reward')[this_local_slice]
            log_progress(
                "chunk_done",
                chunk=int(chunk_idx + 1),
                n_chunks=int(n_chunks),
                elapsed_sec=float(time.monotonic() - chunk_start_time))

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

        log_progress("run_done", n_logged_metrics=int(len(log_data)))
        return log_data

    def transform_action(self, action):
        if self.adapter_checkpoint is not None:
            return action.astype(np.float32)

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
