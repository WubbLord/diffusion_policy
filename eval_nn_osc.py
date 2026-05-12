"""Eval DP via Brian-style NN→OSC adapter (analogous to his eval_inverse_model
but with OSC controller instead of JOINT_POSITION).
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os, pathlib, click, hydra, torch, dill, wandb, json
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def inject_brian_osc_runner(cfg, adapter_path, adapter_obs_keys=None):
    dataset_cfg = cfg.task.get('dataset', None)
    dataset_target = '' if dataset_cfg is None else dataset_cfg.get('_target_', '')
    expected = ('diffusion_policy.dataset.robomimic_replay_joint_delta_lowdim_dataset.'
                'RobomimicReplayJointDeltaLowdimDataset')
    if dataset_target != expected:
        raise RuntimeError(f"Not a joint-delta checkpoint. Got {dataset_target!r}.")

    dataset_type = cfg.task.get('dataset_type', 'ph')
    max_steps = 500 if dataset_type == 'mh' else 400

    # Default to the same obs_keys the policy uses, minus the action target
    if adapter_obs_keys is None:
        adapter_obs_keys = list(cfg.task.obs_keys)

    cfg.task.env_runner = OmegaConf.create({
        '_target_': ('diffusion_policy.env_runner.robomimic_joint_brian_osc_runner.'
                     'RobomimicJointBrianOSCRunner'),
        'dataset_path': cfg.task.dataset_path,
        'obs_keys': list(cfg.task.obs_keys),
        'n_train': 6, 'n_train_vis': 2, 'train_start_idx': 0,
        'n_test': 50, 'n_test_vis': 4, 'test_start_seed': 100000,
        'max_steps': max_steps,
        'n_obs_steps': cfg.n_obs_steps,
        'n_action_steps': cfg.n_action_steps,
        'n_latency_steps': cfg.n_latency_steps,
        'render_hw': [128, 128], 'fps': 10, 'crf': 22,
        'past_action': cfg.get('past_action_visible', False),
        'n_envs': 28,
        'adapter_path': adapter_path,
        'adapter_obs_keys': adapter_obs_keys,
        'command_scale': 1.0,
    })
    print(f"Patched cfg with RobomimicJointBrianOSCRunner (adapter={adapter_path})")


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--adapter', required=True)
def main(checkpoint, output_dir, device, adapter):
    if os.path.exists(output_dir):
        pass  # always overwrite in batch mode
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    inject_brian_osc_runner(cfg, adapter)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device); policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    json_log = {}
    for k, v in runner_log.items():
        if isinstance(v, wandb.sdk.data_types.video.Video):
            json_log[k] = v._path
        else:
            json_log[k] = v
    pathlib.Path(output_dir, 'eval_log.json').write_text(json.dumps(json_log, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
