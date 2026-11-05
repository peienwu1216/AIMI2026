"""Classical EEG baselines and physiology checks for BCI Competition III-IIIb.

Run from anywhere:

    python project/analysis/classical_baselines.py

Everything fitted here (band selection, CSP filters, LDA) uses the training
split only. The test split is touched once, for the final score. Results go to
``results/analysis/classical_baselines.json``.

The script answers three questions the deep models cannot:

1. How far does a standard band-power or CSP pipeline get on this data?
2. When inside the 6 s trial does class information appear?
3. Do the classes show the contralateral mu/beta desynchronisation that motor
   imagery predicts? This is the assumption behind C3/C4 swap augmentation.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import dataloader  # noqa: E402

FS = 125  # Hz, from the official dataset description
TRIALS_PER_SUBJECT = 540
SUBJECTS = ("S4", "X11")
BAND_SETS = {
    "mu": [(8, 13)],
    "beta": [(13, 30)],
    "mu+beta": [(8, 13), (13, 30)],
    "filterbank": [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)],
}
WINDOWS_SEC = [(0, 6), (1, 6), (2, 6), (3, 6), (1, 4), (2, 5)]


def bandpass(x, band):
    """Zero-phase band-pass along the last axis (offline analysis only)."""
    sos = butter(4, band, btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def crop(x, window):
    start, stop = (int(t * FS) for t in window)
    return x[..., start:stop]


def fit_csp(x, y):
    """Return the 2x2 CSP filter matrix for trials shaped (N, 2, T)."""
    covs = []
    for label in (0, 1):
        trials = x[y == label]
        cov = np.einsum("nct,ndt->ncd", trials, trials)
        cov /= np.trace(cov, axis1=1, axis2=2)[:, None, None]
        covs.append(cov.mean(axis=0))
    _, vectors = eigh(covs[0], covs[0] + covs[1])
    return vectors.T


def log_var(x):
    return np.log(x.var(axis=-1) + 1e-12)


def features(x, bands, window, filters=None):
    """Log-variance per band; apply CSP filters when they are provided."""
    columns = []
    for index, band in enumerate(bands):
        filtered = crop(bandpass(x, band), window)
        if filters is not None:
            filtered = np.einsum("fc,nct->nft", filters[index], filtered)
        columns.append(log_var(filtered))
    return np.concatenate(columns, axis=1)


def fit_filters(x, y, bands, window):
    return [fit_csp(crop(bandpass(x, band), window), y) for band in bands]


def fit_predict(x_fit, y_fit, x_eval, bands, window, use_csp):
    filters = fit_filters(x_fit, y_fit, bands, window) if use_csp else None
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(features(x_fit, bands, window, filters), y_fit)
    return classifier.predict(features(x_eval, bands, window, filters))


def cross_validate(x, y, bands, window, use_csp, seed=0):
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = [
        np.mean(fit_predict(x[fit], y[fit], x[held], bands, window, use_csp) == y[held])
        for fit, held in folds.split(x, y)
    ]
    return float(np.mean(scores))


def select_and_score(x_train, y_train, x_test, y_test, use_csp):
    """Pick band set and window by 5-fold CV on train, then score on test."""
    candidates = [
        (cross_validate(x_train, y_train, bands, window, use_csp), name, window)
        for name, bands in BAND_SETS.items()
        for window in WINDOWS_SEC
    ]
    cv_accuracy, name, window = max(candidates, key=lambda item: item[0])
    predictions = fit_predict(x_train, y_train, x_test, BAND_SETS[name], window, use_csp)
    return {
        "bands": name,
        "window_sec": list(window),
        "cv_accuracy": 100 * cv_accuracy,
        "test_accuracy": 100 * float(np.mean(predictions == y_test)),
        "predictions": predictions,
    }


def split_by_subject(x, y):
    for index, name in enumerate(SUBJECTS):
        rows = slice(index * TRIALS_PER_SUBJECT, (index + 1) * TRIALS_PER_SUBJECT)
        yield name, x[rows], y[rows]


def evaluate_pipelines(x_train, y_train, x_test, y_test):
    results = {}
    for pipeline, use_csp in (("bandpower_lda", False), ("csp_lda", True)):
        pooled = select_and_score(x_train, y_train, x_test, y_test, use_csp)
        hits = pooled.pop("predictions") == y_test
        pooled["per_subject_test_accuracy"] = {
            name: 100 * float(hits[index * TRIALS_PER_SUBJECT:(index + 1) * TRIALS_PER_SUBJECT].mean())
            for index, name in enumerate(SUBJECTS)
        }

        per_subject, all_hits = {}, []
        for (name, xs, ys), (_, xe, ye) in zip(
            split_by_subject(x_train, y_train), split_by_subject(x_test, y_test)
        ):
            fitted = select_and_score(xs, ys, xe, ye, use_csp)
            all_hits.append(fitted.pop("predictions") == ye)
            per_subject[name] = fitted
        results[pipeline] = {
            "pooled_model": pooled,
            "subject_specific_models": per_subject,
            "subject_specific_overall_accuracy": 100 * float(np.concatenate(all_hits).mean()),
        }
    return results


def time_resolved_decoding(x_train, y_train, x_test, y_test, width=1.0, step=0.25):
    """Per-subject band-power LDA in a sliding window, scored on test."""
    bands = BAND_SETS["mu+beta"]
    duration = x_train.shape[-1] / FS
    starts = np.arange(0, duration - width + 1e-9, step)
    curves = {"window_center_sec": (starts + width / 2).tolist()}
    for (name, xs, ys), (_, xe, ye) in zip(
        split_by_subject(x_train, y_train), split_by_subject(x_test, y_test)
    ):
        curves[name] = [
            100 * float(np.mean(fit_predict(xs, ys, xe, bands, (s, s + width), False) == ye))
            for s in starts
        ]
    return curves


def lateralization(x, y):
    """Mean log band power of C3 minus C4 for each class, subject and band.

    Motor imagery suppresses mu/beta power over the contralateral hemisphere,
    so the two classes should sit on opposite sides of each other. That class
    separation, not the raw sign, is what C3/C4 swap augmentation relies on.
    """
    table = {}
    for name, xs, ys in split_by_subject(x, y):
        table[name] = {}
        for band_name, band in (("mu", (8, 13)), ("beta", (13, 30))):
            index = log_var(bandpass(xs, band))
            index = index[:, 0] - index[:, 1]
            table[name][band_name] = {
                f"class_{label}": float(index[ys == label].mean()) for label in (0, 1)
            }
            table[name][band_name]["pooled_std"] = float(index.std())
    return table


def main():
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    x_train, x_test = train_data[:, 0], test_data[:, 0]
    y_train, y_test = train_label.astype(int), test_label.astype(int)

    report = {
        "pipelines": evaluate_pipelines(x_train, y_train, x_test, y_test),
        "time_resolved": time_resolved_decoding(x_train, y_train, x_test, y_test),
        "lateralization_train": lateralization(x_train, y_train),
    }

    output_dir = PROJECT_DIR / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classical_baselines.json"
    output_path.write_text(json.dumps(report, indent=2))

    for name, result in report["pipelines"].items():
        pooled = result["pooled_model"]
        print(
            f"{name}: pooled {pooled['test_accuracy']:.2f}% "
            f"({pooled['bands']}, {pooled['window_sec']}s, {pooled['per_subject_test_accuracy']}) | "
            f"subject-specific {result['subject_specific_overall_accuracy']:.2f}%"
        )
    print(json.dumps(report["lateralization_train"], indent=2))
    curves = report["time_resolved"]
    for name in SUBJECTS:
        best = int(np.argmax(curves[name]))
        print(f"{name}: peak {curves[name][best]:.1f}% at {curves['window_center_sec'][best]:.2f}s")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
