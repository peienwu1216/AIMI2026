"""Figures for the Lab2 report, in IEEE two-column conference style.

    python project/analysis/report_figures.py            # every figure
    python project/analysis/report_figures.py fig02 fig04  # selected figures

Figures are drawn at final size (3.5 in single column, 7.16 in double column) and
must not be scaled in the report. They carry no titles, because IEEE figures use
captions, and subfigures are labelled (a), (b), (c). Fonts are embedded as Type 42,
since IEEE PDF eXpress rejects the Type 3 fonts matplotlib uses by default.

Every number comes from run artifacts under ``results/``. Dot plots show each seed
as a small open marker and the mean ± population SD as a filled marker. Accuracy
axes on dot plots do not start at zero, which is why they never use bars.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "analysis"))
sys.path.insert(0, str(PROJECT_DIR))
import dataloader  # noqa: E402
from models import EEGNet  # noqa: E402
from run_analysis import BANDS, FFT_POINTS, FS, spatial_filter_summary, temporal_filter_spectra  # noqa: E402

RESULTS = PROJECT_DIR / "results"
OUTPUT_DIR = PROJECT_DIR / "figures" / "report"

SINGLE, DOUBLE = 3.5, 7.16  # column widths in inches
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"  # Okabe-Ito
GRAY, LIGHT = "#555555", "#DDDDDD"
TARGET = 87.0  # accuracy needed for full performance marks
TEST_FINAL_RUN = "final500-leaky-relu-s113"
FINAL_SEEDS = (17, 42, 113, 511, 2026)


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "axes.linewidth": 0.75,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": LIGHT,
            "grid.linewidth": 0.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
            "savefig.dpi": 600,
        }
    )


def save(figure, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{name}.pdf")
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    plt.close(figure)
    print(f"saved {name}")


def panel(axis, text):
    axis.set_title(text, loc="left", pad=3)


# --- data access -------------------------------------------------------------


def run_dirs(prefix, roots=("raw", "mps", "local")):
    found = []
    for root in roots:
        found += sorted(path for path in (RESULTS / root).glob(f"{prefix}-s*") if path.is_dir())
    return [path for path in found if (path / "summary.json").exists()]


def summaries(prefix, roots=("raw", "mps", "local")):
    return [json.loads((path / "summary.json").read_text()) for path in run_dirs(prefix, roots)]


def last10(prefix):
    values = [row["last_10_test_accuracy_mean"] for row in summaries(prefix)]
    if not values:
        raise ValueError(f"No finished runs for {prefix}")
    return np.array(values)


def peak(prefix):
    rows = summaries(prefix)
    return np.array(
        [row.get("peak_test_accuracy_during_training", row.get("peak_test_accuracy")) for row in rows]
    )


def history(run_dir):
    with (run_dir / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def seed_map(prefix):
    return {row.get("seed") or int(row["run_id"].rsplit("-s", 1)[1]): row for row in summaries(prefix)}


def paired_delta(prefix, baseline="scope-all-full"):
    base = seed_map(baseline)
    target = seed_map(prefix)
    return np.array(
        [
            target[seed]["last_10_test_accuracy_mean"] - base[seed]["last_10_test_accuracy_mean"]
            for seed in sorted(set(base) & set(target))
        ]
    )


# --- drawing helpers ---------------------------------------------------------


def spread(count, width=0.09):
    return np.linspace(-width, width, count) if count > 1 else np.zeros(1)


def dots(axis, position, values, color, marker="o", label=None, horizontal=False, size=3.8):
    """Each seed as a small open marker, mean ± SD as a filled marker."""
    values = np.asarray(values, dtype=float)
    offsets = position + spread(len(values))
    seed_style = dict(
        s=7, facecolors="none", edgecolors=color, linewidths=0.45, marker=marker, zorder=2
    )
    mean, sd = values.mean(), values.std()
    style = dict(fmt=marker, color=color, markersize=size, elinewidth=0.8, capsize=1.6,
                 capthick=0.6, markeredgecolor="white", markeredgewidth=0.3, label=label, zorder=3)
    if horizontal:
        axis.scatter(values, offsets, **seed_style)
        axis.errorbar(mean, position, xerr=sd, **style)
    else:
        axis.scatter(offsets, values, **seed_style)
        axis.errorbar(position, mean, yerr=sd, **style)
    return mean, sd


def legend_key(color, marker, label, linestyle="none"):
    """Legend entry whose marker matches the plotted size."""
    return Line2D([], [], color=color, marker=marker, markersize=3.8, linestyle=linestyle, label=label)


def category_axis(axis, labels, horizontal=False):
    positions = np.arange(len(labels))
    if horizontal:
        axis.set_yticks(positions, labels)
        axis.set_ylim(-0.6, len(labels) - 0.4)
        axis.yaxis.grid(False)
    else:
        axis.set_xticks(positions, labels)
        axis.set_xlim(-0.6, len(labels) - 0.4)
        axis.xaxis.grid(False)
    return positions


LAST10_LABEL = "Last-10 test accuracy (%)"


# --- required by the spec ----------------------------------------------------


def fig02_training_curves():
    """Train/test loss and accuracy of the best EEGNet (spec 2.c.i)."""
    best = history(RESULTS / "raw" / TEST_FINAL_RUN)
    seeds = [history(path) for path in run_dirs("final500-leaky-relu", ("raw",))]
    epochs = best["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.35))
    series = (("train", "Train", BLUE, (0, (3.5, 1.5))), ("test", "Test", VERMILLION, "-"))
    for axis, metric, label in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "accuracy", "Accuracy (%)")):
        for split, name, color, style in series:
            stack = np.stack([run[f"{split}_{metric}"] for run in seeds])
            mean, sd = stack.mean(axis=0), stack.std(axis=0)
            axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
            axis.plot(epochs, best[f"{split}_{metric}"], color=color, linestyle=style, linewidth=0.75, label=name)
        axis.set(xlabel="Epoch", ylabel=label, xlim=(1, epochs[-1]))
    axes[0].set_ylim(0.1, 0.7)
    axes[1].set_ylim(55, 100)
    axes[1].axhline(TARGET, color=GRAY, linewidth=0.7, linestyle=":")
    axes[1].text(130, TARGET + 1.1, "87% target", color=GRAY, fontsize=8)
    top = int(best["test_accuracy"].argmax())
    axes[1].plot(epochs[top], best["test_accuracy"][top], marker="*", color="black", markersize=7, zorder=5,
                 linestyle="none", label=f"Peak {best['test_accuracy'][top]:.2f}% (epoch {int(epochs[top])})")
    band = plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.25, linewidth=0)
    for axis, where in ((axes[0], "upper right"), (axes[1], "lower right")):
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles + [band], labels + ["Mean ± SD, 5 seeds"], loc=where)
    panel(axes[0], "(a) Loss")
    panel(axes[1], "(b) Accuracy")
    save(figure, "fig02_training_curves")


def fig03_confusion():
    """Confusion matrices of the best run, overall and per subject."""
    arrays = np.load(RESULTS / "raw" / TEST_FINAL_RUN / "test_predictions.npz")
    labels, predictions = arrays["labels"], arrays["predictions"]
    slices = (("All", slice(None)), ("S4", slice(0, 540)), ("X11", slice(540, 1080)))
    figure, axes = plt.subplots(1, 3, figsize=(SINGLE, 1.42), sharey=True)
    for index, (axis, (name, rows)) in enumerate(zip(axes, slices)):
        counts = np.array(
            [[np.sum((labels[rows] == t) & (predictions[rows] == p)) for p in (0, 1)] for t in (0, 1)]
        )
        recall = 100 * counts / counts.sum(axis=1, keepdims=True)
        image = axis.imshow(recall, cmap="Blues", vmin=0, vmax=100)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{counts[row, column]}\n{recall[row, column]:.1f}%", ha="center",
                          va="center", fontsize=8, color="white" if recall[row, column] > 55 else "black")
        accuracy = 100 * np.trace(counts) / counts.sum()
        axis.set_xticks((0, 1), ("0", "1"))
        axis.set_yticks((0, 1), ("0", "1"))
        axis.set_xlabel("Predicted class")
        axis.grid(False)
        axis.tick_params(length=0)
        panel(axis, f"({'abc'[index]}) {name}: {accuracy:.2f}%")
    axes[0].set_ylabel("True class")
    bar = figure.colorbar(image, ax=axes, fraction=0.045, pad=0.02)
    bar.set_label("Row %", fontsize=8)
    bar.ax.tick_params(labelsize=8, length=1.5)
    bar.outline.set_linewidth(0.4)
    save(figure, "fig03_confusion_matrix")


def fig04_activation():
    """Seven activations at the spec and tuned baselines."""
    names = (("relu", "ReLU"), ("leaky-relu", "LeakyReLU"), ("gelu", "GELU"), ("mish", "Mish"),
             ("silu", "SiLU"), ("elu", "ELU"), ("selu", "SELU"))
    names = sorted(names, key=lambda item: last10(f"p1-activation-tuned-{item[0]}").mean())
    figure, axis = plt.subplots(figsize=(SINGLE, 2.25))
    positions = category_axis(axis, [label for _, label in names], horizontal=True)
    for position, (key, _) in zip(positions, names):
        dots(axis, position + 0.17, last10(f"p1-activation-tuned-{key}"), VERMILLION, "s", horizontal=True)
        dots(axis, position - 0.17, last10(f"p1-activation-spec-{key}"), BLUE, "o", horizontal=True)
    axis.set(xlabel=LAST10_LABEL, xlim=(70, 90), xticks=range(70, 91, 5))
    axis.legend(
        handles=[
            legend_key(VERMILLION, "s", "Tuned: dropout 0.5, lr $10^{-3}$"),
            legend_key(BLUE, "o", "Spec: dropout 0.25, lr $10^{-2}$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.45, 1.0),
        ncols=2,
        columnspacing=1.0,
        handletextpad=0.2,
    )
    save(figure, "fig04_activation")


def numeric_sweep(axis, keys, xs, color=BLUE, marker="o"):
    means = []
    for key, x in zip(keys, xs):
        values = last10(key)
        step = (max(xs) - min(xs)) / 90
        axis.scatter(x + spread(len(values), step), values, s=7, facecolors="none", edgecolors=color,
                     linewidths=0.45, marker=marker, zorder=2)
        means.append(values.mean())
        axis.errorbar(x, values.mean(), yerr=values.std(), fmt=marker, color=color, markersize=3.8,
                      elinewidth=0.8, capsize=1.6, capthick=0.6, markeredgecolor="white",
                      markeredgewidth=0.3, zorder=3)
    axis.plot(xs, means, color=color, linewidth=0.9, zorder=1)


def fig05_elu_and_slope():
    """ELU alpha and LeakyReLU slope sweeps."""
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.05), sharey=True)
    alphas = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    tags = ("0p1", "0p25", "0p5", "0p75", "1p0", "1p5", "2p0")
    numeric_sweep(axes[0], [f"p1-elu-alpha-a{tag}" for tag in tags], alphas)
    slopes = (0.01, 0.05, 0.1, 0.2, 0.3)
    slope_tags = ("0p01", "0p05", "0p1", "0p2", "0p3")
    numeric_sweep(axes[1], [f"p1-leaky-slope-s{tag}" for tag in slope_tags], slopes, VERMILLION, "s")
    axes[0].set(xlabel=r"ELU $\alpha$", ylabel=LAST10_LABEL, ylim=(70, 90), yticks=range(70, 91, 5), xticks=alphas)
    axes[0].set_xticklabels(["0.1", ".25", "0.5", ".75", "1.0", "1.5", "2.0"])
    axes[1].set(xlabel="LeakyReLU negative slope", xticks=slopes)
    panel(axes[0], r"(a) ELU $\alpha$")
    panel(axes[1], "(b) LeakyReLU slope")
    save(figure, "fig05_elu_alpha_and_slope")


def eval_mode_accuracy(run_dir, fit_x, fit_y, test_x, test_y):
    """Train and test accuracy of the saved checkpoint with dropout switched off."""
    config = json.loads((run_dir / "config.json").read_text())
    model = EEGNet(activation=config["activation"], elu_alpha=config["elu_alpha"],
                   dropout=config["dropout"], leaky_slope=config["leaky_slope"])
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        train = 100 * (model(fit_x).argmax(1) == fit_y).float().mean().item()
        test = 100 * (model(test_x).argmax(1) == test_y).float().mean().item()
    return train, test


def fig06_dropout():
    """Dropout sweep. Train and test share one checkpoint and the eval mode."""
    rates = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75)
    tags = ("0p0", "0p1", "0p25", "0p4", "0p5", "0p6", "0p75")
    train_x, train_y, test_x, test_y = dataloader.read_bci_data()
    indices = np.arange(len(train_y))
    strata = (indices >= 540).astype(np.int64) * 2 + train_y.astype(np.int64)
    fit_rows, _ = train_test_split(indices, test_size=0.2, random_state=113, stratify=strata)
    fit_x = torch.tensor(train_x[fit_rows], dtype=torch.float32)
    fit_y = torch.tensor(train_y[fit_rows].astype(np.int64))
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y.astype(np.int64))

    results = {}
    for rate, tag in zip(rates, tags):
        pairs = [eval_mode_accuracy(path, fit_x, fit_y, test_x, test_y) for path in run_dirs(f"p1-dropout-d{tag}", ("raw",))]
        results[str(rate)] = {"train": [p[0] for p in pairs], "test": [p[1] for p in pairs]}
    (RESULTS / "analysis" / "dropout_eval_mode.json").write_text(json.dumps(results, indent=2))

    figure, axis = plt.subplots(figsize=(SINGLE, 2.15))
    for split, color, marker, style in (("train", BLUE, "o", (0, (3.5, 1.5))), ("test", VERMILLION, "s", "-")):
        means = [np.mean(results[str(rate)][split]) for rate in rates]
        sds = [np.std(results[str(rate)][split]) for rate in rates]
        axis.plot(rates, means, color=color, linewidth=0.9, linestyle=style, zorder=1)
        axis.errorbar(rates, means, yerr=sds, fmt=marker, color=color, markersize=3.8, elinewidth=0.8, capsize=1.6,
                      capthick=0.6, markeredgecolor="white", markeredgewidth=0.3, zorder=3)
    axis.set(xlabel="Dropout rate", ylabel="Accuracy (%)", xticks=rates, ylim=(75, 101), yticks=range(75, 101, 5))
    axis.set_xticklabels(["0", "0.1", ".25", "0.4", "0.5", "0.6", ".75"])
    axis.legend(handles=[legend_key(BLUE, "o", "Train", (0, (3.5, 1.5))), legend_key(VERMILLION, "s", "Test", "-")],
                loc="center right")
    save(figure, "fig06_dropout")


def category_panel(axis, entries, title):
    positions = category_axis(axis, [label for label, _ in entries])
    for position, (_, key) in zip(positions, entries):
        dots(axis, position, last10(key), BLUE)
    panel(axis, title)


def fig07_optimization():
    """Learning rate, batch size, weight decay and optimizer."""
    figure, axes = plt.subplots(1, 4, figsize=(DOUBLE, 1.9), sharey=True, width_ratios=(5, 3, 4, 3))
    category_panel(axes[0], [(r"$3{\times}10^{-4}$", "p2-learning-rate-lr3e-4"), (r"$10^{-3}$", "p2-learning-rate-lr1e-3"),
                             (r"$3{\times}10^{-3}$", "p2-learning-rate-lr3e-3"), (r"$10^{-2}$", "p2-learning-rate-lr1e-2"),
                             (r"$3{\times}10^{-2}$", "p2-learning-rate-lr3e-2")], "(a) Learning rate")
    category_panel(axes[1], [("32", "p2-batch-size-b32"), ("64", "p2-batch-size-b64"), ("128", "p2-batch-size-b128")], "(b) Batch size")
    category_panel(axes[2], [("0", "p2-weight-decay-wd0"), (r"$10^{-4}$", "p2-weight-decay-wd1e-4"),
                             (r"$10^{-3}$", "p2-weight-decay-wd1e-3"), (r"$10^{-2}$", "p2-weight-decay-wd1e-2")], "(c) Weight decay")
    category_panel(axes[3], [("Adam", "p2-optimizer-adam"), ("AdamW", "p2-optimizer-adamw"), ("SGD", "p2-optimizer-sgd")], "(d) Optimizer")
    axes[0].set(ylabel=LAST10_LABEL, ylim=(70, 90), yticks=range(70, 91, 5))
    save(figure, "fig07_optimization")


def fig08_training_duration():
    """Training length, full data."""
    groups = {
        "ReLU": (BLUE, "o", ("p0-final-relu", "final300-relu", "final500-relu")),
        "LeakyReLU": (VERMILLION, "s", ("p0-final-leaky-relu", "final300-leaky_relu", "final500-leaky-relu")),
    }
    epochs = np.array((150, 300, 500))
    figure, axis = plt.subplots(figsize=(SINGLE, 2.1))
    handles = []
    for shift, (name, (color, marker, keys)) in zip((-6, 6), groups.items()):
        for metric, style, fill in ((peak, "-", color), (last10, (0, (3.5, 1.5)), "white")):
            values = [metric(key) for key in keys]
            axis.errorbar(epochs + shift, [v.mean() for v in values], yerr=[v.std() for v in values], color=color,
                          marker=marker, markersize=3.8, markerfacecolor=fill, markeredgecolor=color,
                          markeredgewidth=0.7, linestyle=style, linewidth=0.9, elinewidth=0.7, capsize=1.6,
                          capthick=0.6, zorder=3)
        handles.append(legend_key(color, marker, name))
    handles += [Line2D([], [], color=GRAY, linestyle="-", label="Peak (chosen on test)"),
                Line2D([], [], color=GRAY, linestyle=(0, (3.5, 1.5)), label="Last-10 mean")]
    axis.axhline(TARGET, color=GRAY, linewidth=0.7, linestyle=":")
    axis.text(405, TARGET - 0.06, "87% target", color=GRAY, fontsize=8, ha="center", va="top")
    axis.set(xlabel="Training epochs", ylabel="Test accuracy (%)", xticks=epochs, ylim=(85, 90), xlim=(125, 525))
    axis.legend(handles=handles, loc="upper left", ncols=2, columnspacing=1.0)
    save(figure, "fig08_training_duration")


def paired_modes(axis, entries, title, horizontal=False):
    """Full-data and validation-split results side by side (P3)."""
    positions = category_axis(axis, [label for label, _ in entries], horizontal)
    for position, (_, key) in zip(positions, entries):
        dots(axis, position - 0.17, last10(f"{key}-full"), BLUE, "o", horizontal=horizontal)
        dots(axis, position + 0.17, last10(f"{key}-val"), VERMILLION, "s", horizontal=horizontal)
    panel(axis, title)


def fig09_architecture():
    """Kernel lengths, width and the block type."""
    figure, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.05), sharey=True, width_ratios=(6, 3, 4, 2.4))
    paired_modes(axes[0], [("15", "p3-kernel15"), ("25", "p3-kernel25"), ("33", "p3-kernel33"), ("51", "p3-eegnet-tuned"),
                           ("75", "p3-kernel75"), ("101", "p3-kernel101")], "(a) Temporal kernel (samples)")
    paired_modes(axes[1], [("7", "p3-sepkernel7"), ("15", "p3-eegnet-tuned"), ("31", "p3-sepkernel31")], "(b) Second-block kernel")
    paired_modes(axes[2], [("8/2", "p3-f8d2"), ("16/2", "p3-eegnet-tuned"), ("32/2", "p3-f32d2"), ("16/4", "p3-f16d4")], r"(c) Width $F_1/D$")
    paired_modes(axes[3], [("Spec", "p3-eegnet-tuned"), ("True\nsep.", "p3-true-separable")], "(d) Block type")
    axes[0].set(ylabel=LAST10_LABEL, ylim=(80, 90))
    axes[0].legend(
        handles=[legend_key(BLUE, "o", "All 1,080 trials"), legend_key(VERMILLION, "s", "80% (validation split)")],
        loc="upper left",
        ncols=2,
        columnspacing=1.0,
        handletextpad=0.2,
    )
    save(figure, "fig09_architecture_ablation")


# --- schematics --------------------------------------------------------------

MM = 1 / 25.4


def schematic_axis(width_mm, height_mm):
    """Blank axes measured in millimetres, so spacing can be set exactly."""
    figure = plt.figure(figsize=(width_mm * MM, height_mm * MM), constrained_layout=False)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, width_mm), ylim=(0, height_mm))
    axis.axis("off")
    return figure, axis


def box(axis, x, y, width, height, fill, edge=GRAY, linewidth=0.6):
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=1.0",
                                  facecolor=fill, edgecolor=edge, linewidth=linewidth, zorder=2))


def arrow(axis, start, end, color=GRAY, linewidth=0.8):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=2.2,head_width=1.1",
                                   mutation_scale=1, color=color, linewidth=linewidth, shrinkA=0, shrinkB=0, zorder=3))


def save_schematic(figure, name):
    save_svg = OUTPUT_DIR / f"{name}.svg"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none"}):  # keeps labels editable in Inkscape
        figure.savefig(save_svg)
    save(figure, name)


def fig01_architecture():
    """EEGNet tensor flow, with the layer names used in the spec."""
    figure, axis = schematic_axis(181.9, 45)
    top, height = 37.0, 25.0
    blocks = (
        (0.8, 17, "Input", ("2 channels", "750 samples"), "#EEEEEE", "[B, 1, 2, 750]", ""),
        (27.8, 31, "firstconv", ("Conv2d 1→16, (1, 51)", "BatchNorm"), "#D9E8F5", "[B, 16, 2, 750]", "848 parameters"),
        (68.8, 36, "depthwiseConv", ("Conv2d 16→32, (2, 1)", "groups = 16, BatchNorm", "Activation, AvgPool (1, 4)", "Dropout"), "#CFE8DF", "[B, 32, 1, 187]", "128 parameters"),
        (114.8, 36, "separableConv", ("Conv2d 32→32, (1, 15)", "BatchNorm", "Activation, AvgPool (1, 8)", "Dropout"), "#F7E0CF", "[B, 32, 1, 23]", "15,424 parameters"),
        (160.8, 20.3, "classify", ("Flatten", "Linear 736→2"), "#EEEEEE", "[B, 2]", "1,474 param."),
    )
    for index, (x, width, name, lines, fill, shape, params) in enumerate(blocks):
        box(axis, x, top - height, width, height, fill)
        axis.text(x + width / 2, top - 3.4, name, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        body_top = top - 6.3
        body_center = body_top - (height - 7.3) / 2
        for row, line in enumerate(lines):
            offset = (row - (len(lines) - 1) / 2) * 4.0
            axis.text(x + width / 2, body_center - offset, line, ha="center", va="center", fontsize=8, zorder=4)
        axis.text(x + width / 2, top - height - 3.2, shape, ha="center", va="center", fontsize=8, zorder=4)
        if params:
            axis.text(x + width / 2, top - height - 7.0, params, ha="center", va="center", fontsize=8, color=GRAY, zorder=4)
        if index:
            previous = blocks[index - 1]
            arrow(axis, (previous[0] + previous[1] + 1.5, top - height / 2), (x - 1.5, top - height / 2))
    axis.text(90.95, 41.5, "Total 17,874 parameters. B is the batch size.", ha="center", va="center", fontsize=8, color=GRAY)
    save_schematic(figure, "fig01_eegnet_architecture")


def channel_column(axis, x, y_values, fill, width=5.0, height=4.2):
    for y in y_values:
        axis.add_patch(Rectangle((x, y - height / 2), width, height, facecolor=fill, edgecolor=GRAY, linewidth=0.6, zorder=3))


def links(axis, x0, x1, sources, targets, color):
    for source, target in zip(sources, targets):
        axis.plot((x0, x1), (source, target), color=color, linewidth=0.75, zorder=2)


def fig11_convolution_types():
    """Standard, depthwise and separable convolution, drawn with four channels."""
    figure, axis = schematic_axis(181.9, 54)
    rows = np.array((41.0, 34.0, 27.0, 20.0))
    doubled = np.array((42.75, 39.25, 35.75, 32.25, 28.75, 25.25, 21.75, 18.25))
    left_a, left_b, left_c = 2.0, 62.0, 122.0
    for left, title in ((left_a, "(a) Standard"), (left_b, r"(b) Depthwise, $D = 2$"), (left_c, "(c) Depthwise separable")):
        axis.text(left, 50.5, title, ha="left", va="center", fontsize=8)
        channel_column(axis, left + 2, rows, "#EEEEEE")

    # (a) every output reads every input
    links(axis, left_a + 7, left_a + 46, np.repeat(rows, 4), np.tile(rows, 4), VERMILLION)
    channel_column(axis, left_a + 46, rows, "#F7E0CF")
    # (b) each input feeds its own D outputs
    links(axis, left_b + 7, left_b + 46, np.repeat(rows, 2), doubled, BLUE)
    channel_column(axis, left_b + 46, doubled, "#D9E8F5", height=2.6)
    # (c) one filter per channel, then a 1 x 1 mix
    links(axis, left_c + 7, left_c + 24, rows, rows, BLUE)
    channel_column(axis, left_c + 24, rows, "#D9E8F5")
    links(axis, left_c + 29, left_c + 46, np.repeat(rows, 4), np.tile(rows, 4), VERMILLION)
    channel_column(axis, left_c + 46, rows, "#F7E0CF")

    notes = (
        (left_a, "Every output reads every input.", r"Weights: $C_{in}C_{out}K$", "32 × 32 × 15 = 15,360 (spec block)"),
        (left_b, "Each input has its own $D$ filters.", r"Weights: $C_{in}DK$", "16 × 2 × 2 = 64 (spatial layer)"),
        (left_c, "Filter each channel, then mix (1 × 1).", r"Weights: $C_{in}K + C_{in}C_{out}$", "32 × 15 + 32 × 32 = 1,504"),
    )
    for left, note, formula, numbers in notes:
        for y, text in ((12.8, note), (8.2, formula), (3.6, numbers)):
            axis.text(left + 2, y, text, ha="left", va="center", fontsize=8)
    save_schematic(figure, "fig11_convolution_types")


# --- discussion --------------------------------------------------------------


def fig10_elu_function():
    """ELU and its derivative for several alpha values."""
    x = np.linspace(-4, 2, 601)
    figure, axes = plt.subplots(1, 2, figsize=(SINGLE, 1.65))
    styles = ((0.1, BLUE, "-"), (1.0, VERMILLION, (0, (3.5, 1.5))), (2.0, GREEN, (0, (1, 1))))
    for axis in axes:
        axis.axhline(0, color=GRAY, linewidth=0.4)
        axis.axvline(0, color=GRAY, linewidth=0.4)
    for alpha, color, style in styles:
        y = np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1))
        slope = np.where(x > 0, 1.0, alpha * np.exp(np.minimum(x, 0)))
        axes[0].plot(x, y, color=color, linestyle=style, linewidth=1.1, label=fr"$\alpha={alpha:g}$")
        negative = x <= 0
        axes[1].plot(x[negative], slope[negative], color=color, linestyle=style, linewidth=1.1)
        axes[1].plot(x[~negative], slope[~negative], color=color, linestyle=style, linewidth=1.1)
    axes[0].set(xlabel="$x$", ylabel="ELU$(x)$", xlim=(-4, 2), ylim=(-2.3, 2.1))
    axes[1].set(xlabel="$x$", ylabel=r"ELU$^{\prime}(x)$", xlim=(-4, 2), ylim=(-0.1, 2.2))
    axes[0].legend(loc="upper left")
    panel(axes[0], "(a) Function")
    panel(axes[1], "(b) Derivative")
    save(figure, "fig10_elu_function")


def activation_floor(name, alpha=1.0):
    x = torch.linspace(-30, 5, 70001)
    functions = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu, "mish": F.mish, "selu": F.selu,
                 "elu": lambda value: F.elu(value, alpha=alpha)}
    return float(functions[name](x).min())


def fig12_negative_floor():
    """Accuracy against the lowest output of each activation."""
    others = (("relu", "ReLU"), ("gelu", "GELU"), ("silu", "SiLU"), ("mish", "Mish"), ("selu", "SELU"))
    alphas = ((0.1, "0p1"), (0.25, "0p25"), (0.5, "0p5"), (0.75, "0p75"), (1, "1p0"), (1.5, "1p5"), (2, "2p0"))
    other_points = [(activation_floor(key), last10(f"p1-activation-tuned-{key}"), label) for key, label in others]
    elu_points = [(activation_floor("elu", a), last10(f"p1-elu-alpha-a{tag}"), fr"$\alpha$={a:g}") for a, tag in alphas]
    points = other_points + elu_points
    rho, _ = spearmanr([p[0] for p in points], [p[1].mean() for p in points])

    figure, axis = plt.subplots(figsize=(SINGLE, 2.25))
    elu_sorted = sorted(elu_points, key=lambda p: p[0])
    axis.plot([p[0] for p in elu_sorted], [p[1].mean() for p in elu_sorted], color=BLUE, linewidth=0.9, zorder=1)
    style = dict(markersize=3.8, elinewidth=0.8, capsize=1.6, capthick=0.6, linestyle="none",
                 markeredgecolor="white", markeredgewidth=0.3, zorder=3)
    axis.errorbar([p[0] for p in elu_points], [p[1].mean() for p in elu_points],
                  yerr=[p[1].std() for p in elu_points], marker="o", color=BLUE, label="ELU", **style)
    axis.errorbar([p[0] for p in other_points], [p[1].mean() for p in other_points],
                  yerr=[p[1].std() for p in other_points], marker="s", color=VERMILLION, label="Other activations", **style)
    offsets = {"ReLU": (8, 7), "GELU": (-9, 8), "SiLU": (15, -17), "Mish": (-6, -20), "SELU": (-17, 7),
               r"$\alpha$=0.1": (15, -11), r"$\alpha$=0.5": (-9, 9), r"$\alpha$=0.75": (-16, 6),
               r"$\alpha$=1": (17, -7), r"$\alpha$=1.5": (19, -7), r"$\alpha$=2": (17, -9)}
    for floor, values, label in points:
        if label in offsets:
            axis.annotate(label, (floor, values.mean()), xytext=offsets[label], textcoords="offset points",
                          ha="center", fontsize=8)
    axis.text(0.04, 0.94, fr"Spearman $\rho$ = {rho:.2f}, 12 settings", transform=axis.transAxes, va="top", fontsize=8)
    axis.set(xlabel="Lowest possible activation output", ylabel=LAST10_LABEL, xlim=(-2.25, 0.3), ylim=(70, 90), yticks=range(70, 91, 5))
    axis.legend(loc="lower right")
    save(figure, "fig12_negative_floor")


def fig13_subject_scope():
    """Pooled and subject-specific systems, classical and deep."""
    local = json.loads((RESULTS / "local" / "summary.json").read_text())
    systems = {(row["group"], row["label"]): np.array(row["last_10_test_accuracy_mean"]) for row in local["two_model_systems"]}
    classical = json.loads((RESULTS / "analysis" / "classical_baselines.json").read_text())["pipelines"]
    rows = [
        ("CSP + LDA, shared", classical["csp_lda"]["pooled_model"]["test_accuracy"], VERMILLION),
        ("Band power + LDA, shared", classical["bandpower_lda"]["pooled_model"]["test_accuracy"], VERMILLION),
        ("CSP + LDA, per subject", classical["csp_lda"]["subject_specific_overall_accuracy"], VERMILLION),
        ("Band power + LDA, per subject", classical["bandpower_lda"]["subject_specific_overall_accuracy"], VERMILLION),
        ("EEGNet, shared", last10("scope-all-full"), BLUE),
        ("EEGNet, pre-train + fine-tune", systems[("finetune", "S4+X11-full")], BLUE),
        ("EEGNet, per subject", systems[("scope", "S4+X11-full")], BLUE),
    ]
    figure, axis = plt.subplots(figsize=(SINGLE, 1.95))
    positions = category_axis(axis, [row[0] for row in rows], horizontal=True)
    for position, (_, value, color) in zip(positions, rows):
        if np.ndim(value) == 0:
            axis.plot(value, position, marker="D", color=color, markersize=3.6, markeredgecolor="white",
                      markeredgewidth=0.3, linestyle="none", zorder=3)
        else:
            dots(axis, position, value, color, horizontal=True)
    axis.set(xlabel="Test accuracy (%)", xlim=(55, 92))
    axis.legend(
        handles=[legend_key(BLUE, "o", "EEGNet, last-10, 5 seeds"), legend_key(VERMILLION, "D", "Classical, single fit")],
        loc="lower right",
    )
    save(figure, "fig13_subject_scope")


def delta_panel(axis, entries, title):
    positions = category_axis(axis, [label for label, _ in entries], horizontal=True)
    axis.axvline(0, color=GRAY, linewidth=0.7)
    for position, (_, key) in zip(positions, entries):
        dots(axis, position, paired_delta(key), BLUE, horizontal=True)
    axis.set(xlabel="Change in last-10 accuracy (points)")
    panel(axis, title)


def fig14_normalization_augmentation():
    """Paired changes against the pooled baseline with the same seeds."""
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 1.95), width_ratios=(1, 1.1))
    delta_panel(axes[0], [("Per-channel trial z-score", "normalization-per_trial"),
                          ("Joint trial z-score", "normalization-per_trial_joint"),
                          ("Per-subject scale", "normalization-per_subject")], "(a) Normalization")
    delta_panel(axes[1], [("Noise 0.2", "augment-noise0.2"), ("Noise 0.1", "augment-noise0.1"), ("Noise 0.05", "augment-noise0.05"),
                          ("Shift ±6", "augment-shift6"), ("Shift ±12", "augment-shift12"), ("Shift ±25", "augment-shift25"),
                          ("C3/C4 swap", "stack-swap")], "(b) Augmentation")
    for axis in axes:
        axis.set_xlim(-3.2, 3.2)
    save(figure, "fig14_normalization_augmentation")


def fig15_time_window():
    """Retraining on cropped windows, and a sliding-window LDA."""
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.05), width_ratios=(1.15, 1))
    entries = [("0–6", "scope-all-full"), ("0–5", "window-0-5s"), ("0–4", "stack-w04"), ("1–6", "window-1-6s"), ("1–5", "window-1-5s"),
               ("1–4", "window-1-4s"), ("2–6", "window-2-6s"), ("2–5", "window-2-5s"), ("3–6", "window-3-6s")]
    positions = category_axis(axes[0], [label for label, _ in entries])
    for position, (_, key) in zip(positions, entries):
        dots(axes[0], position, last10(key), BLUE)
    axes[0].set(xlabel="Input window (s from epoch start)", ylabel=LAST10_LABEL, ylim=(64, 90))
    panel(axes[0], "(a) EEGNet retrained on each window")
    curves = json.loads((RESULTS / "analysis" / "classical_baselines.json").read_text())["time_resolved"]
    centers = np.array(curves["window_center_sec"])
    axes[1].axhline(50, color=GRAY, linewidth=0.7, linestyle=":")
    axes[1].text(3.0, 49.2, "Chance", color=GRAY, fontsize=8, ha="center", va="top")
    axes[1].plot(centers, curves["S4"], color=BLUE, marker="o", markersize=2.4, linewidth=0.9, label="S4")
    axes[1].plot(centers, curves["X11"], color=VERMILLION, marker="s", markersize=2.4, linewidth=0.9,
                 linestyle=(0, (3.5, 1.5)), label="X11")
    axes[1].set(xlabel="Center of 1 s window (s)", ylabel="Test accuracy (%)", ylim=(45, 75), xlim=(0.4, 5.6))
    axes[1].legend(loc="upper right")
    panel(axes[1], "(b) Sliding-window band power + LDA")
    save(figure, "fig15_time_window")


def fig16_learned_filters():
    """What the first two layers of the final EEGNet learned."""
    runs = run_dirs("final500-leaky-relu", ("raw",))
    states = [torch.load(path / "test_best.pt", map_location="cpu", weights_only=True)["model_state_dict"] for path in runs]
    temporal = [state["firstconv.0.weight"].numpy() for state in states]
    spectra = [temporal_filter_spectra(weights) for weights in temporal]
    spatial = [spatial_filter_summary(state["depthwiseConv.0.weight"].numpy()) for state in states]
    keys = list(BANDS)
    shares = np.array([[s["mean_band_share"][k] for k in keys] for s in spectra]) * 100
    flat = np.array([(BANDS[k][1] - BANDS[k][0]) / (FS / 2) * 100 for k in keys])
    signal = dataloader.read_bci_data()[0][:, 0].astype(np.float64)
    spectrum = (np.abs(np.fft.rfft(signal - signal.mean(axis=-1, keepdims=True), axis=-1)) ** 2).mean(axis=(0, 1))
    bins = np.fft.rfftfreq(signal.shape[-1], d=1 / FS)
    observed = np.array([spectrum[(bins >= BANDS[k][0]) & (bins < BANDS[k][1])].sum() for k in keys]) / spectrum.sum() * 100
    (RESULTS / "analysis").mkdir(exist_ok=True)
    (RESULTS / "analysis" / "final_model_filters.json").write_text(json.dumps(
        {"runs": [path.name for path in runs], "band_share_percent_mean": dict(zip(keys, shares.mean(axis=0).tolist())),
         "band_share_percent_min": dict(zip(keys, shares.min(axis=0).tolist())),
         "band_share_percent_max": dict(zip(keys, shares.max(axis=0).tolist())),
         "flat_percent": dict(zip(keys, flat.tolist())), "input_signal_percent": dict(zip(keys, observed.tolist())),
         "spatial": spatial}, indent=2))

    best = next(index for index, path in enumerate(runs) if path.name == TEST_FINAL_RUN)
    kernels = temporal[best].reshape(temporal[best].shape[0], -1)
    power = np.abs(np.fft.rfft(kernels, n=FFT_POINTS, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(FFT_POINTS, d=1 / FS)
    keep = frequencies <= 40
    power = power[:, keep] / power[:, keep].max(axis=1, keepdims=True)
    order = np.argsort(frequencies[keep][power.argmax(axis=1)])

    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.0), width_ratios=(1.1, 1))
    image = axes[0].imshow(power[order], aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
                           extent=(0, frequencies[keep][-1], len(order) - 0.5, -0.5))
    for edge in (8, 13, 30):
        axes[0].axvline(edge, color="white", linewidth=0.6, linestyle=(0, (3, 2)))
    for center, name in ((10.5, r"$\mu$"), (21.5, r"$\beta$")):
        axes[0].text(center, 0.1, name, ha="center", va="center", fontsize=8, color="white")
    axes[0].grid(False)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Temporal filter (sorted)", yticks=(0, 5, 10, 15))
    bar = figure.colorbar(image, ax=axes[0], fraction=0.04, pad=0.015)
    bar.set_label("Relative power", fontsize=8)
    bar.ax.tick_params(labelsize=8, length=1.5)
    bar.outline.set_linewidth(0.4)
    panel(axes[0], "(a) Frequency response, best run")

    labels = {"below_4Hz": "< 4 Hz", "theta_4_8": "Theta 4–8", "mu_8_13": "Mu 8–13", "beta_13_30": "Beta 13–30", "above_30Hz": "> 30 Hz"}
    positions = category_axis(axes[1], [labels[k] for k in keys][::-1], horizontal=True)
    for index, position in enumerate(positions[::-1]):
        axes[1].plot((flat[index], shares[:, index].mean()), (position, position), color=LIGHT, linewidth=1.4, zorder=1)
        axes[1].plot(flat[index], position, marker="D", markersize=3.6, markerfacecolor="white",
                     markeredgecolor=GRAY, markeredgewidth=0.7, linestyle="none", zorder=2)
        axes[1].plot(observed[index], position, marker="s", markersize=3.6, color=VERMILLION,
                     markeredgecolor="white", markeredgewidth=0.3, linestyle="none", zorder=2)
        dots(axes[1], position, shares[:, index], BLUE, horizontal=True)
    axes[1].set(xlabel="Share of power (%)", xlim=(0, 60))
    axes[1].legend(
        handles=[Line2D([], [], marker="D", markersize=3.6, markerfacecolor="white", markeredgecolor=GRAY, linestyle="none", label="Flat spectrum"),
                 legend_key(BLUE, "o", "Learned, 5 seeds"), legend_key(VERMILLION, "s", "Input signal")],
        loc="center right",
        bbox_to_anchor=(1.0, 0.62),
    )
    panel(axes[1], "(b) Band share of power")
    save(figure, "fig16_learned_filters")
    print("spatial differential filters:", [s["differential_filters"] for s in spatial])


def fig17_bonus_comparison():
    """Learning rate, activation, and accuracy against model size."""
    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), width_ratios=(3, 3, 3.6), sharey=True)
    rates = ("1e-2", "1e-3", "3e-4")
    positions = category_axis(axes[0], (r"$10^{-2}$", r"$10^{-3}$", r"$3{\times}10^{-4}$"))
    for position, rate in zip(positions, rates):
        dots(axes[0], position - 0.17, last10(f"p4-deep-elu-lr{rate}-full"), BLUE, "o")
        dots(axes[0], position + 0.17, last10(f"p4-shallow-lr{rate}-full"), VERMILLION, "s")
    axes[0].set(xlabel="Learning rate", ylabel=LAST10_LABEL, ylim=(74, 90), yticks=range(75, 91, 5))
    axes[0].legend(handles=[legend_key(BLUE, "o", "DeepConvNet"), legend_key(VERMILLION, "s", "ShallowConvNet")], loc="upper left")
    panel(axes[0], "(a) Learning rate")

    names = (("elu", "ELU"), ("relu", "ReLU"), ("leaky_relu", "LeakyReLU"))
    positions = category_axis(axes[1], [label for _, label in names])
    for position, (key_name, _) in zip(positions, names):
        dots(axes[1], position, last10(f"p4-deep-{key_name}-lr1e-3-full"), BLUE)
    axes[1].set(xlabel="DeepConvNet, lr $10^{-3}$")
    panel(axes[1], "(b) Activation")

    candidates = {
        "DeepConvNet": ["p4-deep-elu-lr1e-2-full", "p4-deep-elu-lr1e-3-full", "p4-deep-elu-lr3e-4-full",
                        "p4-deep-relu-lr1e-3-full", "p4-deep-leaky_relu-lr1e-3-full"],
        "ShallowConvNet": ["p4-shallow-lr1e-2-full", "p4-shallow-lr1e-3-full", "p4-shallow-lr3e-4-full"],
    }
    models = [("EEGNet (spec)", "p3-eegnet-tuned-full", BLUE, "o", (0, 8), "center"), ("True separable", "p3-true-separable-full", BLUE, "^", (0, 9), "center")]
    for name, keys in candidates.items():
        best = max(keys, key=lambda prefix: last10(prefix).mean())
        models.append((name, best, VERMILLION, "s", (6, -3) if name.startswith("Shallow") else (0, 7), "left" if name.startswith("Shallow") else "center"))
    for name, prefix, color, marker, offset, align in models:
        values = last10(prefix)
        count = summaries(prefix)[0]["parameter_count"]
        axes[2].errorbar(count, values.mean(), yerr=values.std(), fmt=marker, color=color, markersize=4.2, elinewidth=0.8,
                         capsize=1.6, capthick=0.6, markeredgecolor="white", markeredgewidth=0.3, zorder=3)
        axes[2].annotate(name, (count, values.mean()), xytext=offset, textcoords="offset points", ha=align, fontsize=8)
        print(f"{name}: {prefix}, {count} parameters, {values.mean():.2f} ± {values.std():.2f}")
    axes[2].set(xscale="log", xlim=(1.5e3, 4e5), xlabel="Parameters (log scale)")
    panel(axes[2], "(c) Accuracy against size")
    save(figure, "fig17_bonus_comparison")


def fig18_bonus_curves():
    """Train and test curves of the best DeepConvNet and ShallowConvNet."""
    figure, axes = plt.subplots(1, 4, figsize=(DOUBLE, 1.95))
    settings = (("DeepConvNet", ["p4-deep-elu-lr1e-2-full", "p4-deep-elu-lr1e-3-full", "p4-deep-elu-lr3e-4-full",
                                 "p4-deep-relu-lr1e-3-full", "p4-deep-leaky_relu-lr1e-3-full"]),
                ("ShallowConvNet", ["p4-shallow-lr1e-2-full", "p4-shallow-lr1e-3-full", "p4-shallow-lr3e-4-full"]))
    series = (("train", "Train", BLUE, (0, (3.5, 1.5))), ("test", "Test", VERMILLION, "-"))
    for column, (name, keys) in enumerate(settings):
        best = max(keys, key=lambda prefix: last10(prefix).mean())
        runs = [history(path) for path in run_dirs(best, ("mps",))]
        epochs = runs[0]["epoch"]
        for axis, metric, label in ((axes[2 * column], "loss", "Loss"), (axes[2 * column + 1], "accuracy", "Accuracy (%)")):
            for split, split_name, color, style in series:
                stack = np.stack([run[f"{split}_{metric}"] for run in runs])
                mean, sd = stack.mean(axis=0), stack.std(axis=0)
                axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
                axis.plot(epochs, mean, color=color, linestyle=style, linewidth=0.85, label=split_name)
            axis.set(xlabel="Epoch", ylabel=label, xlim=(1, epochs[-1]), xticks=(50, 100, 150))
        axes[2 * column].set_ylim(0, 0.8)
        axes[2 * column + 1].set_ylim(50, 100)
        panel(axes[2 * column], f"({'ac'[column]}) {name.replace('ConvNet', '')}: loss")
        panel(axes[2 * column + 1], f"({'bd'[column]}) {name.replace('ConvNet', '')}: accuracy")
        print(name, best)
    axes[1].legend(loc="lower right")
    save(figure, "fig18_bonus_curves")


FIGURES = {
    "fig01": fig01_architecture,
    "fig02": fig02_training_curves,
    "fig03": fig03_confusion,
    "fig04": fig04_activation,
    "fig05": fig05_elu_and_slope,
    "fig06": fig06_dropout,
    "fig07": fig07_optimization,
    "fig08": fig08_training_duration,
    "fig09": fig09_architecture,
    "fig10": fig10_elu_function,
    "fig11": fig11_convolution_types,
    "fig12": fig12_negative_floor,
    "fig13": fig13_subject_scope,
    "fig14": fig14_normalization_augmentation,
    "fig15": fig15_time_window,
    "fig16": fig16_learned_filters,
    "fig17": fig17_bonus_comparison,
    "fig18": fig18_bonus_curves,
}


def main():
    apply_style()
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
