"""Eval a joint-delta checkpoint via the FK->OSC runner with a residual NN adapter.

Same as eval_fk.py but injects the runner with residual_adapter_path set,
so each rollout step does
    osc = clip(FK->OSC(q, dq, grip) + clip(NN(state, dq), ±residual_clip), -1, 1)
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os, pathlib, click, hydra, torch, dill, wandb, json
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def inject_fk_residual_runner(cfg, residual_adapter_path, residual_clip,
                               osc_kp=None, osc_damping_ratio=None,
                               n_action_steps_override=None):
    dataset_cfg = cfg.task.get('dataset', None)
    dataset_target = '' if dataset_cfg is None else dataset_cfg.get('_target_', '')
    joint_dataset_target = (
        'diffusion_policy.dataset.robomimic_replay_joint_delta_lowdim_dataset.'
        'RobomimicReplayJointDeltaLowdimDataset')
    if dataset_target != joint_dataset_target:
        raise RuntimeError(f"Not a joint-delta checkpoint. Got {dataset_target!r}.")

    dataset_type = cfg.task.get('dataset_type', 'ph')
    max_steps = 500 if dataset_type == 'mh' else 400

    joint_pos_keys = list(dataset_cfg.get('joint_pos_keys', ['robot0_joint_pos']))
    n_robots = len(joint_pos_keys)
    if n_robots != 1:
        raise RuntimeError("Residual runner is single-arm only for now.")
    eef_pos_keys = [k.replace('_joint_pos', '_eef_pos') for k in joint_pos_keys]
    eef_quat_keys = [k.replace('_joint_pos', '_eef_quat') for k in joint_pos_keys]

    cfg.task.env_runner = OmegaConf.create({
        '_target_': ('diffusion_policy.env_runner.robomimic_joint_fk_to_eef_runner.'
                     'RobomimicJointFKtoEEFRunner'),
        'dataset_path': cfg.task.dataset_path,
        'obs_keys': list(cfg.task.obs_keys),
        'n_train': 6, 'n_train_vis': 2, 'train_start_idx': 0,
        'n_test': 50, 'n_test_vis': 4, 'test_start_seed': 100000,
        'max_steps': max_steps,
        'n_obs_steps': cfg.n_obs_steps,
        'n_action_steps': (n_action_steps_override if n_action_steps_override is not None
                           else cfg.n_action_steps),
        'n_latency_steps': cfg.n_latency_steps,
        'render_hw': [128, 128], 'fps': 10, 'crf': 22,
        'past_action': cfg.get('past_action_visible', False),
        'n_envs': 28,
        'eef_body_name': 'right_hand',
        'joint_pos_obs_keys': joint_pos_keys,
        'eef_pos_obs_keys': eef_pos_keys,
        'eef_quat_obs_keys': eef_quat_keys,
        'delta_pos_clip': 0.05, 'delta_rot_clip': 0.5,
        'osc_kp_pos': osc_kp, 'osc_damping_ratio': osc_damping_ratio,
        'residual_adapter_path': residual_adapter_path,
        'residual_clip': float(residual_clip),
        'residual_obs_keys': list(cfg.task.obs_keys),
        'residual_device': 'cuda:0',
    })
    print(f"Patched cfg: FK->OSC + residual NN (clip={residual_clip})")


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--residual_adapter', required=True)
@click.option('--residual_clip', type=float, default=0.3)
@click.option('--osc_kp', type=float, default=None)
@click.option('--osc_damping_ratio', type=float, default=None)
def main(checkpoint, output_dir, device, residual_adapter, residual_clip,
          osc_kp, osc_damping_ratio):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    inject_fk_residual_runner(cfg, residual_adapter, residual_clip,
                              osc_kp=osc_kp, osc_damping_ratio=osc_damping_ratio)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    dev = torch.device(device)
    policy.to(dev)
    policy.eval()
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)
    json_log = {}
    for k, v in runner_log.items():
        if isinstance(v, wandb.sdk.data_types.video.Video):
            json_log[k] = v._path
        else:
            json_log[k] = v
    json.dump(json_log, open(os.path.join(output_dir, 'eval_log.json'), 'w'),
              indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
