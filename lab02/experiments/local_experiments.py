"""Experiments that ``main.py`` does not cover, run locally on Apple MPS.

``main.py`` trains one pooled model on both subjects. This script reuses the same
``EEGNet`` and the same validation split, and adds the factors the sweep leaves
out: which subjects a model is trained on, per-subject normalisation, data
augmentation, and the time window fed to the network.

    python project/experiments/local_experiments.py run --groups scope,augment
    python project/experiments/local_experiments.py report

Each run writes ``config.json``, ``history.csv``, ``summary.json``,
``test_predictions.npz`` and ``best.pt`` to ``results/local/<run_id>/``. Finished
runs are skipped, so an interrupted batch can simply be started again.

Test labels never drive training. With ``val_ratio > 0`` the checkpoint is chosen
on validation data. With ``val_ratio == 0`` it is chosen at the test peak, which
follows the course definition of highest testing accuracy and is optimistic.
"""
import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import dataloader  # noqa: E402
from models import EEGNet  # noqa: E402

FS = 125  # Hz
TRIALS_PER_SUBJECT = 540
SUBJECT_IDS = {"S4": 0, "X11": 1}
SPLIT_SEED = 113  # same validation split as main.py
SEEDS = (17, 42, 113, 7, 2026)  # the first three match the main.py sweep
RESULTS_DIR = PROJECT_DIR / "results" / "local"

BASE = {
    "activation": "leaky_relu",
    "leaky_slope": 0.01,
    "dropout": 0.5,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "batch_size": 64,
    "epochs": 150,
    "val_ratio": 0.0,
    "scope": "all",
    "normalization": "none",
    "swap": 0.0,
    "noise": 0.0,
    "shift": 0,
    "window_sec": [0, 6],
    "pretrain_epochs": 0,
    "finetune_lr_scale": 1.0,
}


def variants(group, label_to_overrides, seeds):
    return [
        {**BASE, **overrides, "group": group, "label": label, "seed": seed}
        for label, overrides in label_to_overrides.items()
        for seed in seeds
    ]


def build_groups():
    scope = {
        f"{name}-{mode}": {"scope": name, "val_ratio": ratio}
        for name in ("all", "S4", "X11")
        for mode, ratio in (("full", 0.0), ("val", 0.2))
    }
    finetune = {
        f"{name}-{mode}": {
            "scope": name,
            "val_ratio": ratio,
            "pretrain_epochs": 100,
            "finetune_lr_scale": 0.3,
        }
        for name in ("S4", "X11")
        for mode, ratio in (("full", 0.0), ("val", 0.2))
    }
    return {
        "scope": variants("scope", scope, SEEDS),
        "scope_relu": variants(
            "scope_relu", {name: {"scope": name, "activation": "relu"} for name in SUBJECT_IDS}, SEEDS
        ),
        "finetune": variants("finetune", finetune, SEEDS),
        "normalization": variants(
            "normalization",
            {name: {"normalization": name} for name in ("per_trial", "per_trial_joint", "per_subject")},
            SEEDS,
        ),
        "augment": variants(
            "augment",
            {
                "swap0.5": {"swap": 0.5},
                "noise0.05": {"noise": 0.05},
                "noise0.1": {"noise": 0.1},
                "noise0.2": {"noise": 0.2},
                "shift6": {"shift": 6},
                "shift12": {"shift": 12},
                "shift25": {"shift": 25},
                "noise0.1+shift12": {"noise": 0.1, "shift": 12},
            },
            SEEDS[:3],
        ),
        # The 0-4 s window was found by looking at test accuracy, so "w04-val" and
        # "w06-val" re-check it on validation data before anything relies on it.
        "stack": variants(
            "stack",
            {
                "w04-val": {"window_sec": [0, 4], "val_ratio": 0.2},
                "w06-val": {"val_ratio": 0.2},
                "w04": {"window_sec": [0, 4]},
                "swap": {"swap": 0.5},
                "w04+swap": {"window_sec": [0, 4], "swap": 0.5},
                "w04+swap+e300": {"window_sec": [0, 4], "swap": 0.5, "epochs": 300},
                "S4-w04": {"scope": "S4", "window_sec": [0, 4]},
                "X11-w04": {"scope": "X11", "window_sec": [0, 4]},
            },
            SEEDS,
        ),
        "window": variants(
            "window",
            {
                f"{start}-{stop}s": {"window_sec": [start, stop]}
                for start, stop in ((1, 6), (2, 6), (3, 6), (0, 5), (0, 4), (1, 5), (1, 4), (2, 5))
            },
            SEEDS[:3],
        ),
    }


def normalize(x, subject, method, scales):
    if method == "none":
        return x
    if method == "per_trial":
        centered = x - x.mean(dim=-1, keepdim=True)
        return centered / centered.std(dim=-1, keepdim=True).clamp_min(1e-6)
    if method == "per_trial_joint":
        # One mean and scale per trial, shared by both channels, so the power
        # ratio between C3 and C4 survives. "per_trial" removes that ratio.
        centered = x - x.mean(dim=(2, 3), keepdim=True)
        return centered / centered.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    if method == "per_subject":
        return x / scales[subject].view(-1, 1, 1, 1)
    raise ValueError(f"Unknown normalization: {method}")


def augment(x, y, cfg):
    """Return an augmented copy of one batch; ``x`` is (B, 1, 2, T)."""
    if cfg["swap"] > 0:
        # Mirror C3/C4 and flip the label, since left and right imagery are
        # lateralised in opposite directions.
        mirror = torch.rand(len(x), device=x.device) < cfg["swap"]
        x = torch.where(mirror.view(-1, 1, 1, 1), x.flip(2), x)
        y = torch.where(mirror, 1 - y, y)
    if cfg["shift"] > 0:
        offsets = torch.randint(-cfg["shift"], cfg["shift"] + 1, (len(x), 1), device=x.device)
        steps = torch.arange(x.shape[-1], device=x.device).view(1, -1)
        index = ((steps + offsets) % x.shape[-1]).view(len(x), 1, 1, -1).expand_as(x)
        x = x.gather(3, index)
    if cfg["noise"] > 0:
        x = x + cfg["noise"] * x.std(dim=-1, keepdim=True) * torch.randn_like(x)
    return x, y


def prepare_data(cfg, device):
    train_x, train_y, test_x, test_y = dataloader.read_bci_data()
    subjects = np.repeat(list(SUBJECT_IDS.values()), TRIALS_PER_SUBJECT)
    indices = np.arange(len(train_y))
    labels = train_y.astype(np.int64)
    if cfg["val_ratio"] > 0:
        fit_rows, val_rows = train_test_split(
            indices,
            test_size=cfg["val_ratio"],
            random_state=SPLIT_SEED,
            stratify=subjects * 2 + labels,
        )
    else:
        fit_rows, val_rows = indices, indices[:0]

    scales = torch.tensor(
        [train_x[fit_rows][subjects[fit_rows] == s].std() for s in SUBJECT_IDS.values()],
        dtype=torch.float32,
    )
    start, stop = (int(t * FS) for t in cfg["window_sec"])

    def tensors(x, y, s):
        s = torch.as_tensor(s, dtype=torch.long)
        x = torch.as_tensor(x[..., start:stop].copy(), dtype=torch.float32)
        x = normalize(x, s, cfg["normalization"], scales)
        return x.to(device), torch.as_tensor(y.astype(np.int64)).to(device), s.to(device)

    return {
        "fit": tensors(train_x[fit_rows], train_y[fit_rows], subjects[fit_rows]),
        "val": tensors(train_x[val_rows], train_y[val_rows], subjects[val_rows]),
        "test": tensors(test_x, test_y, subjects),
        "samples": stop - start,
    }


def select_subject(split, scope):
    if scope == "all":
        return split
    keep = split[2] == SUBJECT_IDS[scope]
    return tuple(part[keep] for part in split)


def evaluate(model, x, y):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        return F.cross_entropy(logits, y).item(), logits.argmax(dim=1)


def run_name(cfg):
    return f"{cfg['group']}-{cfg['label']}-s{cfg['seed']}"


def train_one(cfg):
    run_dir = RESULTS_DIR / run_name(cfg)
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text())

    torch.set_num_threads(1)
    device = torch.device(cfg.get("device", "mps"))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    data = prepare_data(cfg, device)
    pooled_x, pooled_y, _ = data["fit"]
    fit_x, fit_y, _ = select_subject(data["fit"], cfg["scope"])
    val_x, val_y, _ = select_subject(data["val"], cfg["scope"])
    test_x, test_y, test_s = select_subject(data["test"], cfg["scope"])

    model = EEGNet(
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        leaky_slope=cfg["leaky_slope"],
        input_samples=data["samples"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    use_val = len(val_y) > 0
    rows = []
    best_score, best_epoch, best_state, best_predictions = -1.0, 0, None, None
    started = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        pretraining = epoch <= cfg["pretrain_epochs"]
        if cfg["pretrain_epochs"] and epoch == cfg["pretrain_epochs"] + 1:
            for group in optimizer.param_groups:
                group["lr"] *= cfg["finetune_lr_scale"]
        x, y = (pooled_x, pooled_y) if pretraining else (fit_x, fit_y)

        model.train()
        order = torch.randperm(len(x), device=device)
        loss_sum, correct, seen = 0.0, 0, 0
        for begin in range(0, len(x), cfg["batch_size"]):
            rows_in_batch = order[begin:begin + cfg["batch_size"]]
            if len(rows_in_batch) < 2:  # BatchNorm cannot use a batch of one
                continue
            batch_x, batch_y = augment(x[rows_in_batch], y[rows_in_batch], cfg)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_y)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            seen += len(batch_y)

        test_loss, predictions = evaluate(model, test_x, test_y)
        hits = predictions == test_y
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": 100 * correct / seen,
            "test_loss": test_loss,
            "test_accuracy": 100 * hits.float().mean().item(),
            "val_accuracy": float("nan"),
        }
        for name, subject in SUBJECT_IDS.items():
            mask = test_s == subject
            row[f"test_{name}"] = (
                100 * hits[mask].float().mean().item() if mask.any() else float("nan")
            )
        if use_val:
            _, val_predictions = evaluate(model, val_x, val_y)
            row["val_accuracy"] = 100 * (val_predictions == val_y).float().mean().item()
        rows.append(row)

        score = row["val_accuracy"] if use_val else row["test_accuracy"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = predictions.cpu().numpy()

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    with (run_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = test_y.cpu().numpy()
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        labels=labels,
        predictions=best_predictions,
        final_predictions=predictions.cpu().numpy(),
    )
    torch.save(
        {
            "model_state_dict": {k: v.cpu() for k, v in best_state.items()},
            "best_epoch": best_epoch,
            "config": cfg,
        },
        run_dir / "best.pt",
    )

    accuracy = np.array([row["test_accuracy"] for row in rows])
    summary = {
        "run_id": run_dir.name,
        "group": cfg["group"],
        "label": cfg["label"],
        "seed": cfg["seed"],
        "scope": cfg["scope"],
        "selection_metric": "val_accuracy" if use_val else "test_accuracy",
        "selected_epoch": best_epoch,
        "test_accuracy_at_selected_checkpoint": 100 * float(np.mean(best_predictions == labels)),
        "peak_test_accuracy": float(accuracy.max()),
        "peak_epoch": int(accuracy.argmax()) + 1,
        "last_10_test_accuracy_mean": float(accuracy[-10:].mean()),
        "final_test_accuracy": float(accuracy[-1]),
        "final_train_accuracy": rows[-1]["train_accuracy"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "seconds": time.time() - started,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def read_curve(run_id):
    with (RESULTS_DIR / run_id / "history.csv").open() as stream:
        return np.array([float(row["test_accuracy"]) for row in csv.DictReader(stream)])


def combine_subject_curves(curve_s4, curve_x11):
    """Overall accuracy of a two-model system, epoch by epoch.

    Both subjects have 540 test trials, so overall accuracy is their mean.
    Taking the peak of this curve picks one shared epoch instead of letting each
    subject choose its own, which would inflate the result further.
    """
    curve = (curve_s4 + curve_x11) / 2
    return {
        "peak_test_accuracy": float(curve.max()),
        "last_10_test_accuracy_mean": float(curve[-10:].mean()),
        "final_test_accuracy": float(curve[-1]),
    }


def describe(values):
    return f"{np.mean(values):6.2f} ± {np.std(values):4.2f}"


def report():
    summaries = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*/summary.json")]
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[(summary["group"], summary["label"])].append(summary)

    metrics = (
        "last_10_test_accuracy_mean",
        "peak_test_accuracy",
        "test_accuracy_at_selected_checkpoint",
    )
    table = []
    for (group, label), runs in sorted(grouped.items()):
        table.append(
            {
                "group": group,
                "label": label,
                "n": len(runs),
                **{m: float(np.mean([r[m] for r in runs])) for m in metrics},
                **{f"{m}_std": float(np.std([r[m] for r in runs])) for m in metrics},
                "best_peak": max(r["peak_test_accuracy"] for r in runs),
            }
        )

    # Pair S4 and X11 models from the same seed into one two-model system.
    combined = []
    for (group, label), runs in sorted(grouped.items()):
        if not label.startswith("S4"):
            continue
        partner_label = label.replace("S4", "X11", 1)
        partners = {r["seed"]: r for r in grouped.get((group, partner_label), [])}
        pairs = [
            combine_subject_curves(read_curve(r["run_id"]), read_curve(partners[r["seed"]]["run_id"]))
            for r in runs
            if r["seed"] in partners
        ]
        if pairs:
            combined.append(
                {
                    "group": group,
                    "label": label.replace("S4", "S4+X11", 1),
                    "n": len(pairs),
                    **{k: [p[k] for p in pairs] for k in pairs[0]},
                }
            )

    print(f"{'group':14} {'label':20} {'n':>2} {'last-10':>14} {'peak':>14} {'selected':>14} {'best':>6}")
    for row in table:
        print(
            f"{row['group']:14} {row['label']:20} {row['n']:2d} "
            + " ".join(
                f"{row[m]:6.2f} ± {row[m + '_std']:4.2f}".rjust(14) for m in metrics
            )
            + f" {row['best_peak']:6.2f}"
        )
    print("\nTwo-model systems (S4 model on S4 test, X11 model on X11 test, shared epoch)")
    for row in combined:
        print(
            f"{row['group']:14} {row['label']:20} {row['n']:2d} "
            f"last-10 {describe(row['last_10_test_accuracy_mean'])}  "
            f"peak {describe(row['peak_test_accuracy'])}  "
            f"best {max(row['peak_test_accuracy']):.2f}"
        )

    output = RESULTS_DIR / "summary.json"
    output.write_text(json.dumps({"per_setting": table, "two_model_systems": combined}, indent=2))
    print(f"\nsaved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--groups", required=True, help="comma-separated group names")
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--device", default="mps")
    commands.add_parser("report")
    args = parser.parse_args()

    if args.command == "report":
        report()
        return

    groups = build_groups()
    configs = [
        {**cfg, "device": args.device}
        for name in args.groups.split(",")
        for cfg in groups[name]
    ]
    print(f"{len(configs)} runs", flush=True)
    with get_context("spawn").Pool(args.workers) as pool:
        for done, summary in enumerate(pool.imap_unordered(train_one, configs), start=1):
            print(
                f"[{done}/{len(configs)}] {summary['run_id']}: "
                f"last10 {summary['last_10_test_accuracy_mean']:.2f} "
                f"peak {summary['peak_test_accuracy']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
