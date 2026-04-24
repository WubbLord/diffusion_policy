from typing import Dict, List
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
    """Robomimic low-dim replay dataset with joint-delta action targets.

    The original Robomimic actions are OSC end-effector commands. This dataset
    replaces the arm action target with q[t + 1] - q[t] computed from low-dim
    joint-position observations. The gripper target is kept from the original
    action vector because it is already a gripper command, not an EEF command.
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
            include_gripper=True,
            gripper_action_indices: List[int] = [-1],
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
        ):
        obs_keys = list(obs_keys)
        joint_pos_keys = list(joint_pos_keys)
        gripper_action_indices = list(gripper_action_indices)

        if include_gripper and len(gripper_action_indices) != len(joint_pos_keys):
            raise ValueError(
                "gripper_action_indices must have one entry per joint_pos_key "
                "when include_gripper=True")

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_joint_delta_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    joint_pos_keys=joint_pos_keys,
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


def _data_to_joint_delta_obs(
        raw_obs,
        raw_actions,
        obs_keys,
        joint_pos_keys,
        include_gripper,
        gripper_action_indices):
    obs_lengths = [raw_obs[key].shape[0] for key in obs_keys]
    joint_lengths = [raw_obs[key].shape[0] for key in joint_pos_keys]
    n_steps = min([raw_actions.shape[0]] + obs_lengths + joint_lengths) - 1

    obs = np.concatenate([
        raw_obs[key][:n_steps] for key in obs_keys
    ], axis=-1).astype(np.float32)

    action_parts = list()
    for key in joint_pos_keys:
        joint_pos = raw_obs[key][:n_steps + 1].astype(np.float32)
        action_parts.append(joint_pos[1:] - joint_pos[:-1])

    if include_gripper:
        action_dim = raw_actions.shape[-1]
        for idx in gripper_action_indices:
            if idx < 0:
                idx = action_dim + idx
            action_parts.append(raw_actions[:n_steps, idx:idx + 1])

    action = np.concatenate(action_parts, axis=-1).astype(np.float32)
    return {
        'obs': obs,
        'action': action,
    }
