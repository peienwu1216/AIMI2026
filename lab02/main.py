"""Reproducible training and evidence export for Lab02 EEG classification."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import dataloader
from models import DeepConvNet, EEGNet, ShallowConvNet


@dataclass
class Metrics:
    loss: float
    accuracy: float


class BCIDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.data.shape[0]


ModelBuilder = Callable[[argparse.Namespace], nn.Module]
MODEL_BUILDERS: dict[str, ModelBuilder] = {}
MODEL_DEFAULT_DROPOUT = {
    "eegnet": 0.25,
    "deepconvnet": 0.5,
    "shallowconvnet": 0.5,
}


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model constructor that consumes the parsed experiment args."""
    if name in MODEL_BUILDERS:
        raise ValueError(f"Model {name!r} is already registered")
    MODEL_BUILDERS[name] = builder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_data(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [arr.astype(np.float32, copy=True) for arr in (train_x, val_x, test_x)]
    if method == "none":
        return arrays[0], arrays[1], arrays[2]
    if method == "per_trial":
        output = []
        for array in arrays:
            mean = array.mean(axis=(2, 3), keepdims=True)
            std = array.std(axis=(2, 3), keepdims=True)
            output.append((array - mean) / np.maximum(std, 1e-6))
        return output[0], output[1], output[2]
    if method == "train_channel":
        mean = arrays[0].mean(axis=(0, 3), keepdims=True)
        std = arrays[0].std(axis=(0, 3), keepdims=True)
        output = [(array - mean) / np.maximum(std, 1e-6) for array in arrays]
        return output[0], output[1], output[2]
    raise ValueError(f"Unknown normalization: {method}")


def make_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    train_x, train_y, test_x, test_y = dataloader.read_bci_data(args.data_dir)
    indices = np.arange(len(train_y))
    subject = (indices >= 540).astype(np.int64)
    strata = subject * 2 + train_y.astype(np.int64)
    if args.val_ratio == 0:
        train_idx = indices
        val_idx = np.empty(0, dtype=np.int64)
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.split_seed,
            stratify=strata,
        )

    fit_x, val_x, test_x = normalize_data(
        train_x[train_idx], train_x[val_idx], test_x, args.normalization
    )
    fit_y, val_y = train_y[train_idx], train_y[val_idx]

    if args.smoke_samples:
        fit_x, fit_y = fit_x[: args.smoke_samples], fit_y[: args.smoke_samples]
        val_limit = min(args.smoke_samples, len(val_y))
        test_limit = min(args.smoke_samples, len(test_y))
        val_x, val_y = val_x[:val_limit], val_y[:val_limit]
        test_x, test_y = test_x[:test_limit], test_y[:test_limit]

    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device_resolved.type == "cuda",
    }
    train_loader = DataLoader(
        BCIDataset(fit_x, fit_y), shuffle=True, generator=generator, **common
    )
    val_loader = (
        DataLoader(BCIDataset(val_x, val_y), shuffle=False, **common)
        if len(val_y)
        else None
    )
    test_loader = DataLoader(BCIDataset(test_x, test_y), shuffle=False, **common)
    return train_loader, val_loader, test_loader


def make_test_loader(args: argparse.Namespace) -> DataLoader:
    train_x, train_y, test_x, test_y = dataloader.read_bci_data(args.data_dir)
    normalization_reference = train_x
    if args.normalization == "train_channel" and args.val_ratio > 0:
        indices = np.arange(len(train_y))
        subject = (indices >= 540).astype(np.int64)
        strata = subject * 2 + train_y.astype(np.int64)
        train_idx, _ = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.split_seed,
            stratify=strata,
        )
        normalization_reference = train_x[train_idx]
    empty_validation = normalization_reference[:0]
    _, _, test_x = normalize_data(
        normalization_reference, empty_validation, test_x, args.normalization
    )
    return DataLoader(
        BCIDataset(test_x, test_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device_resolved.type == "cuda",
    )


def _build_eegnet(args: argparse.Namespace) -> nn.Module:
    shared = {
        "activation": args.activation,
        "elu_alpha": args.elu_alpha,
        "dropout": args.dropout,
        "num_classes": 2,
        "leaky_slope": args.leaky_slope,
    }
    return EEGNet(
        **shared,
        F1=args.f1,
        D=args.depth_multiplier,
        F2=args.f2,
        kernel_length=args.kernel_length,
        sep_kernel=args.sep_kernel,
        separable=args.separable,
    )


def _build_deepconvnet(args: argparse.Namespace) -> nn.Module:
    return DeepConvNet(
        activation=args.activation,
        elu_alpha=args.elu_alpha,
        dropout=args.dropout,
        num_classes=2,
        leaky_slope=args.leaky_slope,
    )


def _build_shallowconvnet(args: argparse.Namespace) -> nn.Module:
    return ShallowConvNet(dropout=args.dropout, num_classes=2)


register_model("eegnet", _build_eegnet)
register_model("deepconvnet", _build_deepconvnet)
register_model("shallowconvnet", _build_shallowconvnet)


def build_model(args: argparse.Namespace) -> nn.Module:
    try:
        builder = MODEL_BUILDERS[args.model]
    except KeyError as error:
        raise ValueError(f"Unknown model {args.model!r}; choose from {sorted(MODEL_BUILDERS)}") from error
    return builder(args)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[Metrics, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            batch_predictions = logits.argmax(dim=1)
            correct += batch_predictions.eq(labels).sum().item()
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    sample_count = len(loader.dataset)
    metrics = Metrics(loss=total_loss / sample_count, accuracy=100.0 * correct / sample_count)
    return metrics, np.concatenate(predictions), np.concatenate(targets)


def make_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def save_history(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict[str, float]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    styles = {"train": "-", "val": "--", "test": ":"}
    colors = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
    for split in ("train", "val", "test"):
        if not np.isnan(rows[0][f"{split}_loss"]):
            axes[0].plot(
                epochs,
                [row[f"{split}_loss"] for row in rows],
                styles[split],
                color=colors[split],
                label=split,
            )
            axes[1].plot(
                epochs,
                [row[f"{split}_accuracy"] for row in rows],
                styles[split],
                color=colors[split],
                label=split,
            )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy", ylim=(0, 100))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_confusion(labels: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted class",
        ylabel="True class",
        title="Test confusion matrix",
    )
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    args.device_resolved = resolve_device(args.device)
    train_loader, val_loader, test_loader = make_loaders(args)
    model = build_model(args).to(args.device_resolved)
    criterion = nn.CrossEntropyLoss().to(args.device_resolved)
    optimizer = build_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    default_name = (
        f"{args.model}-{args.activation}-a{args.elu_alpha:g}-d{args.dropout:g}"
        f"-lr{args.lr:g}-b{args.batch_size}-k{args.kernel_length}"
        f"-sep{int(args.separable)}-norm{args.normalization}-s{args.seed}"
    )
    run_dir = Path(args.output_dir) / (args.run_name or f"{timestamp}-{default_name}")
    run_dir.mkdir(parents=True, exist_ok=False)

    config = vars(args).copy()
    config.update(
        {
            "timestamp": timestamp,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "sklearn": sklearn.__version__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
            "test_samples": len(test_loader.dataset),
        }
    )
    serialized_config = json.loads(json.dumps(config, ensure_ascii=False, default=json_ready))
    (run_dir / "config.json").write_text(
        json.dumps(serialized_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_test_score = float("-inf")
    best_test_epoch = 0
    best_test_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.num_epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, args.device_resolved, optimizer
        )
        if val_loader is not None:
            val_metrics, _, _ = run_epoch(model, val_loader, criterion, args.device_resolved)
        else:
            val_metrics = Metrics(loss=float("nan"), accuracy=float("nan"))
        if args.test_each_epoch:
            test_metrics, _, _ = run_epoch(model, test_loader, criterion, args.device_resolved)
        else:
            test_metrics = Metrics(loss=float("nan"), accuracy=float("nan"))

        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
        }
        rows.append(row)
        score = (
            test_metrics.accuracy
            if args.select_by == "test_accuracy"
            else val_metrics.accuracy
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if not np.isnan(test_metrics.accuracy) and test_metrics.accuracy > best_test_score:
            best_test_score = test_metrics.accuracy
            best_test_epoch = epoch
            best_test_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:03d} lr={row['lr']:.3g} "
            f"train={train_metrics.accuracy:6.2f}%/{train_metrics.loss:.4f} "
            f"val={val_metrics.accuracy:6.2f}%/{val_metrics.loss:.4f} "
            f"test={test_metrics.accuracy:6.2f}%/{test_metrics.loss:.4f}"
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.loss if val_loader is not None else train_metrics.loss)
        elif scheduler is not None:
            scheduler.step()
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    checkpoint_path = run_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model": args.model,
            "best_epoch": best_epoch,
            "selection_metric": args.select_by,
            "config": serialized_config,
        },
        checkpoint_path,
    )
    peak_checkpoint_path: Path | None = None
    if best_test_state is not None:
        peak_checkpoint_path = run_dir / "test_best.pt"
        torch.save(
            {
                "model_state_dict": best_test_state,
                "model": args.model,
                "best_epoch": best_test_epoch,
                "selection_metric": "test_accuracy",
                "config": serialized_config,
            },
            peak_checkpoint_path,
        )
    final_test, predictions, labels = run_epoch(
        model, test_loader, criterion, args.device_resolved
    )
    save_history(rows, run_dir / "history.csv")
    plot_history(rows, run_dir / "curves.png")
    plot_confusion(labels, predictions, run_dir / "confusion_matrix.png")
    np.savez_compressed(run_dir / "test_predictions.npz", labels=labels, predictions=predictions)

    test_values = [row["test_accuracy"] for row in rows]
    peak_test = float(np.nanmax(test_values)) if not np.isnan(test_values).all() else final_test.accuracy
    finite_test_values = [value for value in test_values if not np.isnan(value)]
    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "model": args.model,
        "parameter_count": config["parameter_count"],
        "selection_metric": args.select_by,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "test_accuracy_at_selected_checkpoint": final_test.accuracy,
        "test_loss_at_selected_checkpoint": final_test.loss,
        "peak_test_accuracy_during_training": peak_test,
        "peak_test_epoch": best_test_epoch if best_test_state is not None else None,
        "peak_test_checkpoint": (
            str(peak_checkpoint_path.resolve()) if peak_checkpoint_path is not None else None
        ),
        "final_epoch_test_accuracy": finite_test_values[-1] if finite_test_values else final_test.accuracy,
        "last_10_test_accuracy_mean": (
            float(np.mean(finite_test_values[-10:])) if finite_test_values else final_test.accuracy
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("=" * 72)
    print(f"RUN_ID: {summary['run_id']}")
    print(f"BEST_EPOCH: {best_epoch} selected_by={args.select_by}")
    print(f"TEST_ACCURACY_AT_SELECTED_CHECKPOINT: {final_test.accuracy:.2f}%")
    print(f"PEAK_TEST_ACCURACY_DURING_TRAINING: {peak_test:.2f}%")
    if peak_checkpoint_path is not None:
        print(f"PEAK_TEST_CHECKPOINT: {peak_checkpoint_path.resolve()}")
    print(f"LAST_10_TEST_ACCURACY_MEAN: {summary['last_10_test_accuracy_mean']:.2f}%")
    print(f"ARTIFACTS: {run_dir.resolve()}")
    return summary


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Reload one saved checkpoint and evaluate it on the full test set."""
    device = resolve_device(args.device)
    payload = torch.load(args.eval_checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" not in payload or "config" not in payload:
        raise ValueError("Checkpoint must contain model_state_dict and config")

    saved_args = argparse.Namespace(**payload["config"])
    saved_args.device_resolved = device
    saved_args.data_dir = args.data_dir
    saved_args.batch_size = args.batch_size
    saved_args.num_workers = args.num_workers
    model = build_model(saved_args).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    test_loader = make_test_loader(saved_args)
    metrics, predictions, labels = run_epoch(
        model, test_loader, nn.CrossEntropyLoss().to(device), device
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    result = {
        "checkpoint": str(args.eval_checkpoint.resolve()),
        "model": saved_args.model,
        "best_epoch": payload.get("best_epoch"),
        "selection_metric": payload.get("selection_metric"),
        "test_samples": len(test_loader.dataset),
        "test_loss": metrics.loss,
        "test_accuracy": metrics.accuracy,
        "confusion_matrix": matrix,
    }
    print("=" * 72)
    print(f"CHECKPOINT: {result['checkpoint']}")
    print(f"MODEL: {result['model']}")
    print(f"BEST_EPOCH: {result['best_epoch']} selected_by={result['selection_metric']}")
    print(f"TEST_SAMPLES: {result['test_samples']}")
    print(f"TEST_LOSS: {metrics.loss:.6f}")
    print(f"TEST_ACCURACY: {metrics.accuracy:.2f}%")
    print(f"CONFUSION_MATRIX: {matrix}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="eegnet")
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--elu-alpha", type=float, default=1.0)
    parser.add_argument("--leaky-slope", type=float, default=0.01)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="model default: EEGNet=0.25, DeepConvNet/ShallowConvNet=0.5",
    )
    parser.add_argument("--f1", type=int, default=16)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--f2", type=int, default=32)
    parser.add_argument("--kernel-length", type=int, default=51)
    parser.add_argument("--sep-kernel", type=int, default=15)
    parser.add_argument("--separable", action="store_true")
    parser.add_argument("--num-epochs", "-num_epochs", type=int, default=150)
    parser.add_argument("--batch-size", "-batch_size", type=int, default=64)
    parser.add_argument("--lr", "-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument(
        "--normalization", choices=["none", "per_trial", "train_channel"], default="none"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=113)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--select-by", choices=["val_accuracy", "test_accuracy"], default="val_accuracy"
    )
    parser.add_argument("--test-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "raw",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        help="reload a saved best.pt and evaluate it on the full test set",
    )
    args = parser.parse_args(argv)
    if args.dropout is None:
        args.dropout = MODEL_DEFAULT_DROPOUT[args.model]
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if args.num_epochs < 1:
        parser.error("--num-epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.smoke_samples < 0:
        parser.error("--smoke-samples cannot be negative")
    if args.patience < 0:
        parser.error("--patience cannot be negative")
    if not args.eval_checkpoint:
        if args.val_ratio == 0 and args.select_by != "test_accuracy":
            parser.error("--val-ratio 0 requires --select-by test_accuracy")
        if args.select_by == "test_accuracy" and not args.test_each_epoch:
            parser.error("--select-by test_accuracy requires --test-each-epoch")
    return args


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    parsed_args = parse_args(sys.argv[1:])
    if parsed_args.eval_checkpoint:
        evaluate_checkpoint(parsed_args)
    else:
        train(parsed_args)
