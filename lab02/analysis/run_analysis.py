"""Post-hoc analysis of saved runs from ``main.py``.

    python project/analysis/run_analysis.py RUN_DIR [RUN_DIR ...]

For each run this reports accuracy with a Wilson interval, the split by subject,
and, for EEGNet, what the first two layers learned. With several runs it also
compares each one against the first using an exact McNemar test.

Results are written to ``results/analysis/run_analysis.json``.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, norm

FS = 125  # Hz
TRIALS_PER_SUBJECT = 540
SUBJECTS = ("S4", "X11")
BANDS = {
    "below_4Hz": (0, 4),
    "theta_4_8": (4, 8),
    "mu_8_13": (8, 13),
    "beta_13_30": (13, 30),
    "above_30Hz": (30, FS / 2),
}
FFT_POINTS = 512
PROJECT_DIR = Path(__file__).resolve().parents[1]


def wilson_interval(successes, total, confidence=0.95):
    z = norm.ppf(0.5 + confidence / 2)
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return 100 * float(center - margin), 100 * float(center + margin)


def mcnemar_exact(labels, predictions_a, predictions_b):
    """Exact McNemar test on the trials where exactly one model is right."""
    correct_a = predictions_a == labels
    correct_b = predictions_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    p_value = 1.0 if discordant == 0 else binomtest(only_a, discordant, 0.5).pvalue
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": float(p_value)}


def accuracy_summary(labels, predictions):
    hits = predictions == labels
    low, high = wilson_interval(int(hits.sum()), len(hits))
    summary = {"accuracy": 100 * float(hits.mean()), "ci95": [low, high], "n": int(len(hits))}
    if len(hits) == TRIALS_PER_SUBJECT * len(SUBJECTS):
        summary["per_subject"] = {
            name: 100 * float(hits[i * TRIALS_PER_SUBJECT:(i + 1) * TRIALS_PER_SUBJECT].mean())
            for i, name in enumerate(SUBJECTS)
        }
    summary["per_class_recall"] = {
        f"class_{label}": 100 * float(hits[labels == label].mean()) for label in (0, 1)
    }
    return summary


def temporal_filter_spectra(weights):
    """Share of each temporal kernel's power that falls in each EEG band."""
    kernels = weights.reshape(weights.shape[0], -1)
    power = np.abs(np.fft.rfft(kernels, n=FFT_POINTS, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(FFT_POINTS, d=1 / FS)
    total = power.sum(axis=1, keepdims=True)
    shares = {
        name: (power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1) / total[:, 0])
        for name, (low, high) in BANDS.items()
    }
    peaks = frequencies[power.argmax(axis=1)]
    return {
        "peak_frequency_hz": [round(float(value), 2) for value in peaks],
        "mean_band_share": {name: float(value.mean()) for name, value in shares.items()},
        "dominant_band_count": {
            name: int(count)
            for name, count in zip(
                shares,
                np.bincount(np.stack(list(shares.values())).argmax(axis=0), minlength=len(shares)),
            )
        },
    }


def spatial_filter_summary(weights):
    """Sort depthwise filters into C3-C4 difference and C3+C4 sum patterns.

    Opposite signs act as a lateralised (differential) spatial filter. Matching
    signs average the two hemispheres.
    """
    pairs = weights.reshape(weights.shape[0], 2)
    differential = np.sign(pairs[:, 0]) != np.sign(pairs[:, 1])
    return {
        "filters": int(len(pairs)),
        "differential_filters": int(differential.sum()),
        "common_mode_filters": int((~differential).sum()),
        "mean_abs_c3": float(np.abs(pairs[:, 0]).mean()),
        "mean_abs_c4": float(np.abs(pairs[:, 1]).mean()),
    }


def load_state(run_dir):
    # Checkpoints here are produced locally by main.py, so a full load is safe.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    return checkpoint["model_state_dict"]


def analyze_run(run_dir):
    arrays = np.load(run_dir / "test_predictions.npz")
    labels, predictions = arrays["labels"], arrays["predictions"]
    result = {"run": run_dir.name, **accuracy_summary(labels, predictions)}
    state = load_state(run_dir)
    if "firstconv.0.weight" in state:
        result["temporal_filters"] = temporal_filter_spectra(state["firstconv.0.weight"].numpy())
        result["spatial_filters"] = spatial_filter_summary(state["depthwiseConv.0.weight"].numpy())
    return result, labels, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "results" / "analysis" / "run_analysis.json"
    )
    args = parser.parse_args()

    results, comparisons = [], []
    reference = None
    for run_dir in args.runs:
        result, labels, predictions = analyze_run(run_dir)
        results.append(result)
        if reference is None:
            reference = (run_dir.name, labels, predictions)
        else:
            if not np.array_equal(reference[1], labels):
                raise ValueError(f"{run_dir} was scored on different test labels")
            comparisons.append(
                {
                    "first": reference[0],
                    "second": run_dir.name,
                    **mcnemar_exact(labels, reference[2], predictions),
                }
            )

    report = {"runs": results, "mcnemar_vs_first": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
