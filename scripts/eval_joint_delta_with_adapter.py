#!/usr/bin/env python
import json
import os
import pathlib
import sys

import click
import dill
import hydra
import torch
import wandb
from omegaconf import OmegaConf, open_dict


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option("--checkpoint", required=True)
@click.option("--adapter-checkpoint", required=True)
@click.option("--output-dir", required=True)
@click.option("--device", default="cuda:0")
@click.option("--adapter-device", default="cpu")
@click.option("--n-test", type=int, default=None)
@click.option("--n-test-vis", type=int, default=None)
@click.option("--n-train", type=int, default=None)
@click.option("--n-train-vis", type=int, default=None)
@click.option("--n-envs", type=int, default=None)
@click.option("--max-steps", type=int, default=None)
@click.option("--num-inference-steps", type=int, default=None)
@click.option("--overwrite", is_flag=True)
def main(
        checkpoint,
        adapter_checkpoint,
        output_dir,
        device,
        adapter_device,
        n_test,
        n_test_vis,
        n_train,
        n_train_vis,
        n_envs,
        max_steps,
        num_inference_steps,
        overwrite):
    output_dir = pathlib.Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    if cfg.task.env_runner is None:
        raise RuntimeError("Checkpoint config has no task.env_runner to patch.")

    adapter_payload = torch.load(
        open(adapter_checkpoint, "rb"), map_location="cpu")
    adapter_meta = adapter_payload.get("dataset_metadata", {})
    joint_delta_scale = adapter_meta.get("joint_delta_scale", [0.25] * 7)

    with open_dict(cfg.task.env_runner):
        cfg.task.env_runner._target_ = (
            "diffusion_policy.env_runner.robomimic_joint_lowdim_runner."
            "RobomimicJointLowdimRunner")
        cfg.task.env_runner.adapter_checkpoint = adapter_checkpoint
        cfg.task.env_runner.adapter_device = adapter_device
        cfg.task.env_runner.adapter_obs_keys = adapter_meta.get("obs_keys", [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
        ])
        cfg.task.env_runner.joint_delta_scale = joint_delta_scale

        for key, value in {
            "n_test": n_test,
            "n_test_vis": n_test_vis,
            "n_train": n_train,
            "n_train_vis": n_train_vis,
            "n_envs": n_envs,
            "max_steps": max_steps,
        }.items():
            if value is not None:
                cfg.task.env_runner[key] = value

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)

    policy.to(torch.device(device))
    policy.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(output_dir))
    try:
        runner_log = env_runner.run(policy)
    finally:
        close_fn = getattr(env_runner.env, "close", None)
        if close_fn is not None:
            close_fn()

    json_log = {
        "checkpoint": checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "device": device,
        "adapter_device": adapter_device,
        "num_inference_steps": int(policy.num_inference_steps),
        "env_runner": OmegaConf.to_container(cfg.task.env_runner, resolve=True),
        "runner_log": {},
    }
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log["runner_log"][key] = value._path
        else:
            try:
                json.dumps(value)
                json_log["runner_log"][key] = value
            except TypeError:
                json_log["runner_log"][key] = str(value)

    with open(output_dir / "eval_log.json", "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    print(json.dumps(json_log, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
