import json
import pathlib
import sys
from typing import Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


DEFAULT_OBS_KEYS = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_csv(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def get_demo_names(dataset_path: str) -> List[str]:
    with h5py.File(dataset_path, "r") as f:
        names = sorted(
            f["data"].keys(),
            key=lambda name: int(name.split("_")[-1]),
        )
    return names


def get_obs_dim(dataset_path: str, obs_keys: Sequence[str]) -> int:
    with h5py.File(dataset_path, "r") as f:
        demo = f["data"][get_demo_names(dataset_path)[0]]
        return int(sum(demo["obs"][key].shape[-1] for key in obs_keys))


def build_state_features(obs_group, obs_keys: Sequence[str], t: int) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs_group[key][t], dtype=np.float32) for key in obs_keys],
        axis=0,
    )


def stack_state_features(obs_group, obs_keys: Sequence[str]) -> np.ndarray:
    parts = [np.asarray(obs_group[key][:], dtype=np.float32) for key in obs_keys]
    return np.concatenate(parts, axis=-1)


def infer_joint_keys(obs_keys: Iterable[str]) -> List[str]:
    return [key for key in obs_keys if key.endswith("_joint_pos")]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def expand_scale(scale_values: Sequence[float], dim: int) -> np.ndarray:
    if len(scale_values) == 1:
        return np.full((dim,), float(scale_values[0]), dtype=np.float32)
    scale = np.asarray(scale_values, dtype=np.float32)
    if scale.shape != (dim,):
        raise ValueError(f"Expected one scale or {dim} per-joint scales, got {scale.shape}.")
    if np.any(scale <= 0):
        raise ValueError("Joint command scales must be positive.")
    return scale


def compute_normalizer(x: np.ndarray, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return {"mean": mean, "std": std}


def normalize(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (x - stats["mean"]) / stats["std"]


def denormalize(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    return x * stats["std"] + stats["mean"]


class InverseControllerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=(512, 512, 512),
        activation="silu",
        layer_norm=True,
        residual=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.residual = residual

        if activation == "relu":
            act_cls = nn.ReLU
        elif activation == "gelu":
            act_cls = nn.GELU
        elif activation == "silu":
            act_cls = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation {activation!r}")

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_inverse_checkpoint(path: str, device="cpu"):
    payload = torch.load(path, map_location=device)
    cfg = payload["model_config"]
    model = InverseControllerMLP(**cfg)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    normalizer = {}
    for name, stats in payload["normalizer"].items():
        normalizer[name] = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in stats.items()
        }
    return payload, model, normalizer


def predict_command(model, normalizer, state, desired_delta, command_scale):
    x = np.concatenate([state, desired_delta], axis=-1).astype(np.float32)
    with torch.no_grad():
        tx = torch.as_tensor(x, dtype=torch.float32, device=next(model.parameters()).device)
        tx = normalize(tx, normalizer["input"])
        pred = model(tx)
        pred = denormalize(pred, normalizer["command"])
    command = pred.detach().cpu().numpy()
    return np.clip(command, -command_scale, command_scale)
