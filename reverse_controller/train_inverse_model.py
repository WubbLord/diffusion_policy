import argparse
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from reverse_controller.common import (
    InverseControllerMLP,
    compute_normalizer,
    load_json,
    normalize,
    save_json,
)


def parse_hidden_dims(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def demo_index_from_path(path):
    return int(path.stem.split("_")[-1])


def list_shards(dataset_dir, demo_start=None, demo_end=None, max_shards=None):
    shard_paths = sorted(
        (pathlib.Path(dataset_dir) / "shards").glob("demo_*.npz"),
        key=demo_index_from_path,
    )
    if demo_start is not None:
        shard_paths = [p for p in shard_paths if demo_index_from_path(p) >= demo_start]
    if demo_end is not None:
        shard_paths = [p for p in shard_paths if demo_index_from_path(p) < demo_end]
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]
    return shard_paths


def load_shards(dataset_dir, demo_start=None, demo_end=None, max_shards=None, desc="load shards"):
    shard_paths = list_shards(
        dataset_dir,
        demo_start=demo_start,
        demo_end=demo_end,
        max_shards=max_shards,
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found in {dataset_dir}/shards for "
            f"demo_start={demo_start}, demo_end={demo_end}")

    states = []
    desired = []
    commands = []
    demo_delta = []
    demo_ids = []
    for path in tqdm.tqdm(shard_paths, desc=desc):
        data = np.load(path)
        states.append(data["state"].astype(np.float32))
        desired.append(data["desired_delta"].astype(np.float32))
        commands.append(data["command"].astype(np.float32))
        demo_delta.append(data["demo_delta"].astype(np.float32))
        demo_ids.append(demo_index_from_path(path))

    state = np.concatenate(states, axis=0)
    desired_delta = np.concatenate(desired, axis=0)
    command = np.concatenate(commands, axis=0)
    demo_delta = np.concatenate(demo_delta, axis=0)
    x = np.concatenate([state, desired_delta], axis=-1).astype(np.float32)
    return {
        "x": x,
        "state": state,
        "desired_delta": desired_delta,
        "command": command,
        "demo_delta": demo_delta,
        "n_shards": len(shard_paths),
        "demo_ids": demo_ids,
    }


def make_split(n, val_ratio, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def evaluate(model, loader, input_stats, command_stats, device):
    model.eval()
    losses = []
    mae = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(normalize(xb, input_stats))
            target = normalize(yb, command_stats)
            loss = torch.nn.functional.mse_loss(pred, target)
            pred_command = pred * command_stats["std"] + command_stats["mean"]
            losses.append(loss.item())
            mae.append(torch.mean(torch.abs(pred_command - yb)).item())
    return {
        "loss": float(np.mean(losses)),
        "command_mae": float(np.mean(mae)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dims", default="512,512,512")
    parser.add_argument("--activation", default="silu", choices=["silu", "relu", "gelu"])
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--train-demo-start", type=int, default=None)
    parser.add_argument("--train-demo-end", type=int, default=None)
    parser.add_argument("--val-demo-start", type=int, default=None)
    parser.add_argument("--val-demo-end", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metadata = load_json(pathlib.Path(args.dataset_dir) / "metadata.json")
    command_scale = np.asarray(metadata["joint_delta_scale"], dtype=np.float32)

    train_arrays = load_shards(
        args.dataset_dir,
        demo_start=args.train_demo_start,
        demo_end=args.train_demo_end,
        max_shards=args.max_shards,
        desc="load train shards",
    )
    if args.val_demo_start is not None or args.val_demo_end is not None:
        val_arrays = load_shards(
            args.dataset_dir,
            demo_start=args.val_demo_start,
            demo_end=args.val_demo_end,
            max_shards=None,
            desc="load val shards",
        )
        x_train = train_arrays["x"]
        command_train = train_arrays["command"]
        x_val = val_arrays["x"]
        command_val = val_arrays["command"]
        split_mode = "heldout_demo"
    else:
        x = train_arrays["x"]
        command = train_arrays["command"]
        train_idx, val_idx = make_split(len(x), args.val_ratio, args.seed)
        x_train = x[train_idx]
        command_train = command[train_idx]
        x_val = x[val_idx]
        command_val = command[val_idx]
        val_arrays = None
        split_mode = "random_row"

    input_stats_np = compute_normalizer(x_train)
    command_stats_np = compute_normalizer(command_train)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    input_stats = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in input_stats_np.items()
    }
    command_stats = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in command_stats_np.items()
    }

    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(command_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.as_tensor(x_val, dtype=torch.float32),
        torch.as_tensor(command_val, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model_config = {
        "input_dim": int(x_train.shape[-1]),
        "output_dim": int(command_train.shape[-1]),
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "activation": args.activation,
        "layer_norm": not args.no_layer_norm,
        "residual": False,
    }
    model = InverseControllerMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = {
        "args": vars(args),
        "dataset_metadata": metadata,
        "split_mode": split_mode,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "train_demo_ids": train_arrays["demo_ids"],
        "val_demo_ids": [] if val_arrays is None else val_arrays["demo_ids"],
        "n_train_shards": int(train_arrays["n_shards"]),
        "n_val_shards": 0 if val_arrays is None else int(val_arrays["n_shards"]),
        "model_config": model_config,
        "device": str(device),
    }
    save_json(output_dir / "config.json", run_config)

    history = []
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_mae = []
        start_time = time.time()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(normalize(xb, input_stats))
            target = normalize(yb, command_stats)
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_command = pred * command_stats["std"] + command_stats["mean"]
                pred_command = torch.clamp(
                    pred_command,
                    min=torch.as_tensor(-command_scale, device=device),
                    max=torch.as_tensor(command_scale, device=device),
                )
                mae = torch.mean(torch.abs(pred_command - yb)).item()
            train_losses.append(loss.item())
            train_mae.append(mae)
            pbar.set_postfix(loss=np.mean(train_losses), mae=np.mean(train_mae))

        val_metrics = evaluate(model, val_loader, input_stats, command_stats, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_command_mae": float(np.mean(train_mae)),
            "val_loss": val_metrics["loss"],
            "val_command_mae": val_metrics["command_mae"],
            "seconds": float(time.time() - start_time),
        }
        history.append(row)
        save_json(output_dir / "history.json", history)
        print(row, flush=True)

        payload = {
            "model": model.state_dict(),
            "model_config": model_config,
            "normalizer": {
                "input": input_stats_np,
                "command": command_stats_np,
            },
            "dataset_metadata": metadata,
            "command_scale": command_scale,
            "epoch": epoch,
            "history": history,
        }
        torch.save(payload, latest_path)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(payload, best_path)

    hist = load_json(output_dir / "history.json")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot([x["epoch"] for x in hist], [x["train_loss"] for x in hist], label="train")
    ax.plot([x["epoch"] for x in hist], [x["val_loss"] for x in hist], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized command MSE")
    ax.legend()
    fig.savefig(output_dir / "loss.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
