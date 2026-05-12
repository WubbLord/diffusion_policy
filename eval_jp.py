"""Eval a joint-delta checkpoint via JOINT_POSITION runner with adjustable kp.

Mirrors eval.py's patch_legacy_joint_delta_runner but lets you override the
controller kp (default 50, way too low — causes the slow-arm drift). Pass
--kp 300 to test the higher-gain variant.

Usage:
    python eval_jp.py --checkpoint <path> --output_dir <path> --kp 300
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


def inject_jp_runner(cfg, kp=None, damping_ratio=None,
                     joint_delta_scale=None):
    dataset_cfg = cfg.task.get('dataset', None)
    dataset_target = '' if dataset_cfg is None else dataset_cfg.get('_target_', '')
    expected = ('diffusion_policy.dataset.robomimic_replay_joint_delta_lowdim_dataset.'
                'RobomimicReplayJointDeltaLowdimDataset')
    if dataset_target != expected:
        raise RuntimeError(f"Not a joint-delta checkpoint. Got {dataset_target!r}.")

    joint_pos_keys = list(dataset_cfg.get('joint_pos_keys', []))
    n_robots = len(joint_pos_keys)
    if n_robots <= 0:
        raise RuntimeError("Cannot infer robot count.")
    gripper_indices = list(dataset_cfg.get('gripper_action_indices', []))
    gripper_dims = [1 for _ in range(n_robots)]
    if len(gripper_indices) not in {0, n_robots}:
        raise RuntimeError("Gripper layout mismatch.")

    dataset_type = cfg.task.get('dataset_type', 'ph')
    max_steps = 500 if dataset_type == 'mh' else 400

    runner_cfg = {
        '_target_': ('diffusion_policy.env_runner.robomimic_joint_lowdim_runner.'
                     'RobomimicJointLowdimRunner'),
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
        'n_robots': n_robots,
        'joint_dims': [7 for _ in range(n_robots)],
        'gripper_dims': gripper_dims,
        # default per-joint scale derived from can/ph max (no clipping)
        'joint_delta_scale': (joint_delta_scale if joint_delta_scale is not None
                              else [0.035, 0.07, 0.045, 0.125, 0.085, 0.135, 0.105]),
        'joint_action_mode': 'delta',
        'input_action_layout': 'joints_then_grippers',
    }
    if kp is not None:
        runner_cfg['controller_kp'] = float(kp)
    if damping_ratio is not None:
        runner_cfg['controller_damping_ratio'] = float(damping_ratio)
    cfg.task.env_runner = OmegaConf.create(runner_cfg)
    print(f"Patched checkpoint with JOINT_POSITION runner (kp={kp}, damping={damping_ratio}).")


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--kp', type=float, default=None)
@click.option('--damping_ratio', type=float, default=None)
def main(checkpoint, output_dir, device, kp, damping_ratio):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    inject_jp_runner(cfg, kp=kp, damping_ratio=damping_ratio)
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
        cfg.task.env_runner, output_dir=output_dir)
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
