"""Eval a joint-delta checkpoint via the new FK→OSC runner.

Mirrors eval.py but injects RobomimicJointFKtoEEFRunner instead of the
default JOINT_POSITION runner. Use this for any joint-delta checkpoint
whose task.env_runner is null.

Usage:
    python eval_fk.py --checkpoint <path> --output_dir <path>
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def inject_fk_runner(cfg, osc_kp=None, osc_damping_ratio=None):
    """Replace cfg.task.env_runner with RobomimicJointFKtoEEFRunner config.

    Works regardless of whether the checkpoint was trained with env_runner
    set or null. Always overwrites.
    """
    dataset_cfg = cfg.task.get('dataset', None)
    dataset_target = '' if dataset_cfg is None else dataset_cfg.get('_target_', '')
    joint_dataset_target = (
        'diffusion_policy.dataset.robomimic_replay_joint_delta_lowdim_dataset.'
        'RobomimicReplayJointDeltaLowdimDataset')
    if dataset_target != joint_dataset_target:
        raise RuntimeError(
            "FK runner only supports joint-delta checkpoints "
            f"(dataset target = {dataset_target!r}).")

    joint_action_mode = dataset_cfg.get('joint_action_mode', 'delta')
    if joint_action_mode != 'delta':
        raise RuntimeError(
            f"FK runner only supports joint_action_mode='delta', got "
            f"{joint_action_mode!r}.")

    dataset_type = cfg.task.get('dataset_type', 'ph')
    max_steps = 500 if dataset_type == 'mh' else 400

    # Infer n_robots from dataset's joint_pos_keys.
    joint_pos_keys = list(dataset_cfg.get('joint_pos_keys', ['robot0_joint_pos']))
    n_robots = len(joint_pos_keys)
    eef_pos_keys = [k.replace('_joint_pos', '_eef_pos') for k in joint_pos_keys]
    eef_quat_keys = [k.replace('_joint_pos', '_eef_quat') for k in joint_pos_keys]
    cfg.task.env_runner = OmegaConf.create({
        '_target_': (
            'diffusion_policy.env_runner.robomimic_joint_fk_to_eef_runner.'
            'RobomimicJointFKtoEEFRunner'),
        'dataset_path': cfg.task.dataset_path,
        'obs_keys': list(cfg.task.obs_keys),
        'n_train': 6,
        'n_train_vis': 2,
        'train_start_idx': 0,
        'n_test': 50,
        'n_test_vis': 4,
        'test_start_seed': 100000,
        'max_steps': max_steps,
        'n_obs_steps': cfg.n_obs_steps,
        'n_action_steps': cfg.n_action_steps,
        'n_latency_steps': cfg.n_latency_steps,
        'render_hw': [128, 128],
        'fps': 10,
        'crf': 22,
        'past_action': cfg.get('past_action_visible', False),
        'n_envs': 28,
        'eef_body_name': 'right_hand',
        'joint_pos_obs_keys': joint_pos_keys,
        'eef_pos_obs_keys': eef_pos_keys,
        'eef_quat_obs_keys': eef_quat_keys,
        'delta_pos_clip': 0.05,
        'delta_rot_clip': 0.5,
        'osc_kp_pos': osc_kp,
        'osc_damping_ratio': osc_damping_ratio,
    })
    print("Patched joint-delta checkpoint with RobomimicJointFKtoEEFRunner.")


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--osc_kp', type=float, default=None)
@click.option('--osc_damping_ratio', type=float, default=None)
def main(checkpoint, output_dir, device, osc_kp, osc_damping_ratio):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    inject_fk_runner(cfg, osc_kp=osc_kp, osc_damping_ratio=osc_damping_ratio)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)

    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
