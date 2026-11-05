import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import local_experiments as lx  # noqa: E402

NO_AUGMENT = {"swap": 0.0, "noise": 0.0, "shift": 0}


def batch(size=8, samples=30):
    x = torch.arange(size * 2 * samples, dtype=torch.float32).view(size, 1, 2, samples)
    return x, torch.tensor([0, 1] * (size // 2))


def test_no_augmentation_is_identity() -> None:
    x, y = batch()
    out_x, out_y = lx.augment(x, y, NO_AUGMENT)
    assert torch.equal(out_x, x)
    assert torch.equal(out_y, y)


def test_full_swap_mirrors_channels_and_flips_labels() -> None:
    x, y = batch()
    out_x, out_y = lx.augment(x, y, {**NO_AUGMENT, "swap": 1.1})
    assert torch.equal(out_x[:, :, 0], x[:, :, 1])
    assert torch.equal(out_x[:, :, 1], x[:, :, 0])
    assert torch.equal(out_y, 1 - y)


def test_shift_is_a_circular_roll_of_each_trial() -> None:
    torch.manual_seed(0)
    x, y = batch()
    out_x, out_y = lx.augment(x, y, {**NO_AUGMENT, "shift": 5})
    assert torch.equal(out_y, y)
    assert not torch.equal(out_x, x)
    for original, shifted in zip(x, out_x):
        rolls = [torch.roll(original, k, -1) for k in range(-5, 6)]
        assert any(torch.equal(shifted, r) for r in rolls)


def test_noise_keeps_the_input_tensor_unchanged() -> None:
    x, y = batch()
    before = x.clone()
    out_x, _ = lx.augment(x, y, {**NO_AUGMENT, "noise": 0.1})
    assert torch.equal(x, before)
    assert not torch.equal(out_x, x)


def test_per_trial_normalization_gives_unit_scale_per_channel() -> None:
    x = torch.randn(4, 1, 2, 100) * 7 + 3
    out = lx.normalize(x, None, "per_trial", None)
    assert out.mean(dim=-1).abs().max() < 1e-4
    assert (out.std(dim=-1) - 1).abs().max() < 1e-4


def test_joint_normalization_keeps_the_channel_power_ratio() -> None:
    x = torch.randn(4, 1, 2, 2000)
    x[:, :, 1] *= 3
    joint = lx.normalize(x, None, "per_trial_joint", None)
    separate = lx.normalize(x, None, "per_trial", None)
    ratio = joint[:, :, 1].std(dim=-1) / joint[:, :, 0].std(dim=-1)
    assert (ratio - 3).abs().max() < 0.3
    assert (separate[:, :, 1].std(dim=-1) / separate[:, :, 0].std(dim=-1) - 1).abs().max() < 1e-3


def test_per_subject_normalization_uses_each_subjects_scale() -> None:
    x = torch.ones(2, 1, 2, 4)
    out = lx.normalize(x, torch.tensor([0, 1]), "per_subject", torch.tensor([2.0, 4.0]))
    assert out[0].unique().item() == 0.5
    assert out[1].unique().item() == 0.25


def test_select_subject_keeps_only_that_subjects_rows() -> None:
    subjects = torch.tensor([0, 0, 1])
    split = (torch.arange(3), torch.arange(3), subjects)
    assert lx.select_subject(split, "X11")[0].tolist() == [2]
    assert lx.select_subject(split, "all") is split


def test_combined_curve_uses_one_shared_epoch() -> None:
    s4 = np.array([90.0, 50.0] * 10)
    x11 = np.array([50.0, 90.0] * 10)
    # Each subject peaks at 90 on different epochs, so the system never exceeds 70.
    assert lx.combine_subject_curves(s4, x11)["peak_test_accuracy"] == pytest.approx(70.0)


def test_val_split_matches_main_and_stays_out_of_fit_rows() -> None:
    data = lx.prepare_data({**lx.BASE, "val_ratio": 0.2}, torch.device("cpu"))
    assert len(data["fit"][1]) == 864
    assert len(data["val"][1]) == 216
    assert len(data["test"][1]) == 1080
    for subject in (0, 1):
        for label in (0, 1):
            rows = (data["val"][2] == subject) & (data["val"][1] == label)
            assert rows.sum().item() == 54


def test_window_sets_sample_count_and_model_accepts_it() -> None:
    data = lx.prepare_data({**lx.BASE, "window_sec": [1, 5]}, torch.device("cpu"))
    assert data["samples"] == 500
    assert data["fit"][0].shape[-1] == 500
    model = lx.EEGNet(input_samples=500)
    assert model(data["fit"][0][:4]).shape == (4, 2)


def test_every_run_name_is_unique() -> None:
    names = [lx.run_name(c) for configs in lx.build_groups().values() for c in configs]
    assert len(names) == len(set(names))
