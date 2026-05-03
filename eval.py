"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
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


def patch_legacy_joint_delta_runner(cfg):
    if cfg.task.env_runner is not None:
        return

    dataset_cfg = cfg.task.get('dataset', None)
    dataset_target = '' if dataset_cfg is None else dataset_cfg.get('_target_', '')
    joint_dataset_target = (
        'diffusion_policy.dataset.robomimic_replay_joint_delta_lowdim_dataset.'
        'RobomimicReplayJointDeltaLowdimDataset')
    if dataset_target != joint_dataset_target:
        raise RuntimeError(
            "Checkpoint config has task.env_runner=null, so eval.py does not "
            "know how to run environment rollouts for this policy.")

    joint_action_mode = dataset_cfg.get('joint_action_mode', 'delta')
    if joint_action_mode != 'delta':
        raise RuntimeError(
            "Checkpoint config has task.env_runner=null and uses "
            f"joint_action_mode={joint_action_mode!r}. The built-in fallback "
            "only supports joint_action_mode='delta'.")

    joint_pos_keys = list(dataset_cfg.get('joint_pos_keys', []))
    n_robots = len(joint_pos_keys)
    if n_robots <= 0:
        raise RuntimeError(
            "Cannot infer robot count for legacy joint-delta eval because "
            "task.dataset.joint_pos_keys is empty.")

    gripper_indices = list(dataset_cfg.get('gripper_action_indices', []))
    gripper_dims = [1 for _ in range(n_robots)]
    if len(gripper_indices) not in {0, n_robots}:
        raise RuntimeError(
            "Cannot infer gripper layout for legacy joint-delta eval because "
            f"there are {len(gripper_indices)} gripper indices for "
            f"{n_robots} robots.")

    dataset_type = cfg.task.get('dataset_type', 'ph')
    max_steps = 500 if dataset_type == 'mh' else 400
    cfg.task.env_runner = OmegaConf.create({
        '_target_': (
            'diffusion_policy.env_runner.robomimic_joint_lowdim_runner.'
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
        'joint_delta_scale': 0.05,
        'joint_action_mode': 'delta',
        'input_action_layout': 'joints_then_grippers',
    })
    print("Patched legacy joint-delta checkpoint with RobomimicJointLowdimRunner.")

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    patch_legacy_joint_delta_runner(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
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
