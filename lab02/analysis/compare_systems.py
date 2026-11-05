"""Paired comparison of the pooled model and the two-model (per-subject) system.

    python project/analysis/compare_systems.py

Uses final-epoch predictions, so test accuracy plays no part in choosing the
checkpoint. For each seed, the S4 and X11 models are joined into one set of 1080
predictions and compared with the pooled model from the same seed.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_analysis import mcnemar_exact, wilson_interval  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DIR = PROJECT_DIR / "results" / "local"
SEEDS = (17, 42, 113, 7, 2026)


def final_predictions(run_id):
    arrays = np.load(LOCAL_DIR / run_id / "test_predictions.npz")
    return arrays["labels"], arrays["final_predictions"]


def two_model_predictions(group, s4_label, x11_label, seed):
    labels_s4, predictions_s4 = final_predictions(f"{group}-{s4_label}-s{seed}")
    labels_x11, predictions_x11 = final_predictions(f"{group}-{x11_label}-s{seed}")
    return np.concatenate([labels_s4, labels_x11]), np.concatenate([predictions_s4, predictions_x11])


def majority_vote(predictions):
    """Majority of an odd number of models, as hard votes over two classes."""
    return (np.mean(predictions, axis=0) > 0.5).astype(int)


def main():
    per_seed = []
    pooled_all, two_model_all = [], []
    labels = None
    for seed in SEEDS:
        labels, pooled = final_predictions(f"scope-all-full-s{seed}")
        joined_labels, two_model = two_model_predictions("scope", "S4-full", "X11-full", seed)
        if not np.array_equal(labels, joined_labels):
            raise ValueError("Pooled and two-model test labels are in different orders")
        pooled_all.append(pooled)
        two_model_all.append(two_model)
        per_seed.append(
            {
                "seed": seed,
                "pooled_accuracy": 100 * float(np.mean(pooled == labels)),
                "two_model_accuracy": 100 * float(np.mean(two_model == labels)),
                **mcnemar_exact(labels, pooled, two_model),
            }
        )

    votes = {
        "pooled_5_seed_vote": majority_vote(pooled_all),
        "two_model_5_seed_vote": majority_vote(two_model_all),
    }
    ensembles = {}
    for name, predictions in votes.items():
        hits = int(np.sum(predictions == labels))
        ensembles[name] = {
            "accuracy": 100 * hits / len(labels),
            "ci95": list(wilson_interval(hits, len(labels))),
        }

    report = {
        "per_seed_final_epoch": per_seed,
        "ensembles": ensembles,
        "mcnemar_ensembles": mcnemar_exact(
            labels, votes["pooled_5_seed_vote"], votes["two_model_5_seed_vote"]
        ),
    }
    output = PROJECT_DIR / "results" / "analysis" / "compare_systems.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
