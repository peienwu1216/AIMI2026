import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_analysis import (  # noqa: E402
    FS,
    accuracy_summary,
    mcnemar_exact,
    spatial_filter_summary,
    temporal_filter_spectra,
    wilson_interval,
)


def test_wilson_interval_matches_reference_value() -> None:
    # Reference from the closed form: 940/1080 gives roughly 84.9 to 88.9.
    low, high = wilson_interval(940, 1080)
    assert low == pytest.approx(84.9, abs=0.1)
    assert high == pytest.approx(88.9, abs=0.1)


def test_mcnemar_identical_models_are_not_different() -> None:
    labels = np.array([0, 1, 0, 1])
    result = mcnemar_exact(labels, labels, labels)
    assert result == {"only_first_correct": 0, "only_second_correct": 0, "p_value": 1.0}


def test_mcnemar_counts_discordant_trials() -> None:
    labels = np.zeros(20, dtype=int)
    first = np.zeros(20, dtype=int)
    second = np.zeros(20, dtype=int)
    second[:10] = 1  # first is right on these ten, second is wrong
    result = mcnemar_exact(labels, first, second)
    assert result["only_first_correct"] == 10
    assert result["only_second_correct"] == 0
    assert result["p_value"] == pytest.approx(2 * 0.5**10)


def test_accuracy_summary_splits_subjects_in_file_order() -> None:
    labels = np.tile([0, 1], 540)
    predictions = labels.copy()
    predictions[540:] = 1 - predictions[540:]  # every X11 trial is wrong
    summary = accuracy_summary(labels, predictions)
    assert summary["accuracy"] == pytest.approx(50.0)
    assert summary["per_subject"] == {"S4": 100.0, "X11": 0.0}


def test_temporal_spectra_find_a_10_hz_kernel_in_the_mu_band() -> None:
    time = np.arange(51) / FS
    kernel = (np.sin(2 * np.pi * 10 * time) * np.hanning(51)).reshape(1, 1, 1, 51)
    spectra = temporal_filter_spectra(kernel)
    assert spectra["peak_frequency_hz"][0] == pytest.approx(10, abs=0.5)
    assert spectra["dominant_band_count"]["mu_8_13"] == 1
    assert sum(spectra["mean_band_share"].values()) == pytest.approx(1.0)


def test_spatial_summary_separates_difference_from_sum_filters() -> None:
    weights = np.array([1.0, -1.0, 1.0, 1.0, -2.0, 0.5]).reshape(3, 1, 2, 1)
    summary = spatial_filter_summary(weights)
    assert summary["differential_filters"] == 2
    assert summary["common_mode_filters"] == 1
