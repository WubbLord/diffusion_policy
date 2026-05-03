from typing import Dict, List, Optional
import copy

import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import normalizer_from_stat


class RobomimicReplayJointDeltaLowdimDataset(BaseLowdimDataset):
    """Robomimic low-dim replay dataset with joint-space action targets.

    The original Robomimic actions are OSC end-effector commands. This dataset
    replaces the arm action target with labels derived from low-dim joint
    observations. By default the label is q[t + 1] - q[t]. The gripper target
    is kept from the original action vector because it is already a gripper
    command, not an EEF command.
    """

    def __init__(
            self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
                'robot0_joint_pos'],
            joint_pos_keys: List[str] = ['robot0_joint_pos'],
            joint_vel_keys: Optional[List[str]] = None,
            joint_action_mode: str = 'delta',
            include_gripper=True,
            gripper_action_indices: List[int] = [-1],
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
        ):
        obs_keys = list(obs_keys)
        joint_pos_keys = list(joint_pos_keys)
        joint_vel_keys = None if joint_vel_keys is None else list(joint_vel_keys)
        gripper_action_indices = list(gripper_action_indices)

        valid_modes = {'delta', 'next_pos', 'velocity'}
        if joint_action_mode not in valid_modes:
            raise ValueError(
                f"joint_action_mode must be one of {sorted(valid_modes)}, "
                f"got {joint_action_mode!r}")
        if joint_action_mode == 'velocity':
            if joint_vel_keys is None:
                joint_vel_keys = [
                    key.replace('joint_pos', 'joint_vel')
                    for key in joint_pos_keys
                ]
            if len(joint_vel_keys) != len(joint_pos_keys):
                raise ValueError(
                    "joint_vel_keys must have one entry per joint_pos_key "
                    "when joint_action_mode='velocity'")

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_joint_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    joint_pos_keys=joint_pos_keys,
                    joint_vel_keys=joint_vel_keys,
                    joint_action_mode=joint_action_mode,
                    include_gripper=include_gripper,
                    gripper_action_indices=gripper_action_indices)
                if episode['action'].shape[0] > 0:
                    replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.joint_action_mode = joint_action_mode

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer['action']))
        normalizer['obs'] = normalizer_from_stat(
            array_to_stats(self.replay_buffer['obs']))
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        return dict_apply(data, torch.from_numpy)


class RobomimicReplayJointLowdimDataset(RobomimicReplayJointDeltaLowdimDataset):
    """Name alias for configs that use non-delta joint_action_mode options."""


def _data_to_joint_obs(
        raw_obs,
        raw_actions,
        obs_keys,
        joint_pos_keys,
        joint_vel_keys,
        joint_action_mode,
        include_gripper,
        gripper_action_indices):
    obs_lengths = [raw_obs[key].shape[0] for key in obs_keys]
    joint_lengths = [raw_obs[key].shape[0] for key in joint_pos_keys]
    if joint_action_mode == 'velocity':
        joint_lengths.extend(raw_obs[key].shape[0] for key in joint_vel_keys)
        n_steps = min([raw_actions.shape[0]] + obs_lengths + joint_lengths)
    else:
        n_steps = min([raw_actions.shape[0]] + obs_lengths + joint_lengths) - 1

    obs = np.concatenate([
        raw_obs[key][:n_steps] for key in obs_keys
    ], axis=-1).astype(np.float32)

    action_parts = list()
    if joint_action_mode == 'velocity':
        for key in joint_vel_keys:
            action_parts.append(raw_obs[key][:n_steps].astype(np.float32))
    else:
        for key in joint_pos_keys:
            joint_pos = raw_obs[key][:n_steps + 1].astype(np.float32)
            if joint_action_mode == 'delta':
                action_parts.append(joint_pos[1:] - joint_pos[:-1])
            elif joint_action_mode == 'next_pos':
                action_parts.append(joint_pos[1:])

    if include_gripper:
        action_dim = raw_actions.shape[-1]
        for idx in gripper_action_indices:
            if idx < 0:
                idx = action_dim + idx
            if idx < 0 or idx >= action_dim:
                raise IndexError(
                    f"gripper action index {idx} is out of bounds for "
                    f"raw action dimension {action_dim}")
            action_parts.append(raw_actions[:n_steps, idx:idx + 1])

    action = np.concatenate(action_parts, axis=-1).astype(np.float32)
    return {
        'obs': obs,
        'action': action,
    }


def _data_to_joint_delta_obs(
        raw_obs,
        raw_actions,
        obs_keys,
        joint_pos_keys,
        include_gripper,
        gripper_action_indices):
    return _data_to_joint_obs(
        raw_obs=raw_obs,
        raw_actions=raw_actions,
        obs_keys=obs_keys,
        joint_pos_keys=joint_pos_keys,
        joint_vel_keys=None,
        joint_action_mode='delta',
        include_gripper=include_gripper,
        gripper_action_indices=gripper_action_indices)
