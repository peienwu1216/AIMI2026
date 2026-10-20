"""Generate reproducible IEEE-style learning curves and test figures.

The raster figures are exported at their final publication size and 600 dpi.
Matching PDF/SVG vector files are written to ``figures/vector``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np


MODELS = ("resnet18", "resnet50")
MODEL_LABELS = {"resnet18": "ResNet18", "resnet50": "ResNet50"}
MODEL_STYLE = {
    "resnet18": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "resnet50": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
}

# IEEE final-size dimensions: 3.5 in single column, 7.16 in double column.
IEEE_SINGLE = (3.5, 2.55)
IEEE_MATRIX = (3.5, 2.95)
IEEE_SUMMARY = (3.5, 2.30)
IEEE_DOUBLE = (7.16, 4.85)

# All limits are explicit and percentage-labelled. The local scales make the
# high-performance region legible without hiding the actual displayed range.
CURVE_LIMITS = {
    ("train", "accuracy"): (0.91, 1.001, 0.02),
    ("train", "f1"): (0.94, 1.001, 0.01),
    ("val", "accuracy"): (0.91, 1.001, 0.02),
    ("val", "f1"): (0.94, 1.001, 0.01),
}


def set_ieee_style() -> None:
    """Apply a compact, grayscale-safe IEEE journal style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_json(path: str | Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_results(runs_dir: str | Path) -> dict:
    result = {}
    for name in MODELS:
        folder = Path(runs_dir) / name
        result[name] = {
            "history": read_json(folder / "history.json"),
            "test": read_json(folder / "test_metrics.json"),
        }
    return result


def _export(fig: plt.Figure, figures_dir: str | Path, stem: str) -> None:
    """Save a 600-dpi PNG plus editable PDF and SVG versions."""
    figures_dir = Path(figures_dir)
    vector_dir = figures_dir / "vector"
    figures_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=600)
    fig.savefig(vector_dir / f"{stem}.pdf")
    fig.savefig(vector_dir / f"{stem}.svg")


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.55)
    ax.set_axisbelow(True)


def _plot_curve_axis(
    ax: plt.Axes,
    results: dict,
    phase: str,
    metric: str,
    title: str,
) -> None:
    for name in MODELS:
        history = results[name]["history"]
        epochs = [entry["epoch"] for entry in history]
        scores = [entry[phase][metric] for entry in history]
        style = MODEL_STYLE[name]
        ax.plot(
            epochs,
            scores,
            label=MODEL_LABELS[name],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.75,
            markevery=1,
        )

    low, high, tick_step = CURVE_LIMITS[(phase, metric)]
    last_epoch = max(len(results[name]["history"]) for name in MODELS)
    ax.set_xlim(0.5, last_epoch + 0.5)
    ax.set_ylim(low, high)
    ax.set_xticks([1] + list(range(5, last_epoch + 1, 5)))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else "F1-score (%)")
    ax.set_title(title, loc="left", pad=4)
    ax.legend(loc="lower right", handlelength=2.3)
    _style_axis(ax)


def plot_curves(results: dict, figures_dir: str | Path) -> None:
    panels = [
        ("train", "accuracy", "Training accuracy", "train_accuracy"),
        ("train", "f1", "Training F1-score", "train_f1"),
        ("val", "accuracy", "Validation accuracy", "val_accuracy"),
        ("val", "f1", "Validation F1-score", "val_f1"),
    ]
    for phase, metric, title, stem in panels:
        fig, ax = plt.subplots(figsize=IEEE_SINGLE, layout="constrained")
        _plot_curve_axis(ax, results, phase, metric, title)
        _export(fig, figures_dir, stem)
        plt.close(fig)

    # A double-column, four-panel version for direct journal-style reuse.
    fig, axes = plt.subplots(2, 2, figsize=IEEE_DOUBLE, layout="constrained")
    for ax, (phase, metric, title, _) in zip(axes.flat, panels):
        _plot_curve_axis(ax, results, phase, metric, title)
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.text(
            -0.18,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
        )
    _export(fig, figures_dir, "learning_curves_ieee")
    plt.close(fig)


def plot_confusion(results: dict, figures_dir: str | Path) -> None:
    vmax = max(np.asarray(res["test"]["confusion_matrix"]).max() for res in results.values())
    for name in MODELS:
        cm = np.asarray(results[name]["test"]["confusion_matrix"], dtype=int)
        row_total = cm.sum(axis=1, keepdims=True)
        row_percent = cm / row_total * 100

        fig, ax = plt.subplots(figsize=IEEE_MATRIX)
        im = ax.imshow(cm, cmap="cividis", vmin=0, vmax=vmax, aspect="equal")
        for i in range(2):
            for j in range(2):
                red, green, blue, _alpha = im.cmap(im.norm(cm[i, j]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "black" if luminance > 0.55 else "white"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({row_percent[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=color,
                )
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{MODEL_LABELS[name]} test confusion matrix", loc="left", pad=4)
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
        cbar.set_label("Count", fontsize=7.5)
        cbar.ax.tick_params(labelsize=6.8, width=0.5, length=2.5)
        fig.subplots_adjust(left=0.23, right=0.86, bottom=0.19, top=0.84)
        _export(fig, figures_dir, f"{name}_confusion_matrix")
        plt.close(fig)


def plot_test_summary(results: dict, figures_dir: str | Path) -> None:
    """Render the saved JSON results as clean, auditable output panels."""
    for name in MODELS:
        metrics = results[name]["test"]
        rows = [
            ("Accuracy", f'{metrics["accuracy"] * 100:.2f}%'),
            ("F1-score", f'{metrics["f1"] * 100:.2f}%'),
            ("Precision", f'{metrics["precision"] * 100:.2f}%'),
            ("Recall", f'{metrics["recall"] * 100:.2f}%'),
            ("Specificity", f'{metrics.get("specificity", float("nan")) * 100:.2f}%'),
            ("Threshold", f'{metrics.get("decision_threshold", 0.5):.2f}'),
            ("Test samples", f'{metrics["support"]:,}'),
        ]

        fig, ax = plt.subplots(figsize=IEEE_SUMMARY, layout="constrained")
        ax.axis("off")
        ax.set_title(f"{MODEL_LABELS[name]} final test output", loc="left", pad=5)
        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            colLoc="left",
            cellLoc="left",
            colWidths=[0.62, 0.30],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.6)
        table.scale(1.0, 1.18)
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor("#707070")
            cell.set_linewidth(0.35)
            cell.set_facecolor("white")
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_linewidth(0.7)
            elif row in (1, 2):
                cell.set_facecolor("#EFEFEF")
                cell.set_text_props(weight="bold")
        _export(fig, figures_dir, f"{name}_test_output")
        plt.close(fig)


def create_figures(runs_dir: str | Path = "runs", figures_dir: str | Path = "figures") -> dict:
    set_ieee_style()
    results = load_results(runs_dir)
    plot_curves(results, figures_dir)
    plot_confusion(results, figures_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    create_figures(args.runs_dir, args.figures_dir)
    print(f"IEEE figures saved under {args.figures_dir}/")
    print(f"Editable PDF/SVG files saved under {args.figures_dir}/vector/")


if __name__ == "__main__":
    main()
