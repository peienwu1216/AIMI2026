import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_main_grid as grid  # noqa: E402

import main  # noqa: E402

ALL_JOBS = grid.build_jobs(["p4", "p3"])


def test_every_job_is_accepted_by_the_main_parser() -> None:
    for _, args in ALL_JOBS:
        main.parse_args(args)


def test_every_job_builds_a_model() -> None:
    for name, args in ALL_JOBS:
        parsed = main.parse_args(args)
        assert main.build_model(parsed) is not None, name


def test_run_names_are_unique_and_match_the_flag() -> None:
    names = [name for name, _ in ALL_JOBS]
    assert len(names) == len(set(names))
    for name, args in ALL_JOBS:
        assert args[args.index("--run-name") + 1] == name


def test_counts_cover_both_modes_and_three_seeds() -> None:
    assert len(grid.build_jobs(["p4"])) == 8 * 2 * 3
    assert len(grid.build_jobs(["p3"])) == 12 * 2 * 3


@pytest.mark.parametrize(
    ("label", "f1", "depth", "f2"),
    [("f8d2", 8, 2, 16), ("f32d2", 32, 2, 64), ("f16d4", 16, 4, 64)],
)
def test_width_variants_keep_f2_equal_to_f1_times_depth(label, f1, depth, f2) -> None:
    parsed = main.parse_args(grid.architecture_settings()[label])
    assert (parsed.f1, parsed.depth_multiplier, parsed.f2) == (f1, depth, f2)


def test_final_recipe_trains_300_epochs_on_all_trials_with_p0_seeds() -> None:
    jobs = grid.build_jobs(["final300"])
    assert len(jobs) == 10
    for _, args in jobs:
        parsed = main.parse_args(args)
        assert parsed.num_epochs == 300
        assert parsed.val_ratio == 0
        assert parsed.select_by == "test_accuracy"
        assert parsed.lr == pytest.approx(1e-3)
        assert parsed.dropout == pytest.approx(0.5)
    assert {main.parse_args(args).seed for _, args in jobs} == {17, 42, 113, 511, 2026}


def test_labels_keep_only_the_named_settings() -> None:
    jobs = grid.build_jobs(["p3"], {"true-separable", "f8d2"})
    assert len(jobs) == 2 * 2 * 3
    assert all("true-separable" in name or "f8d2" in name for name, _ in jobs)


def test_output_never_targets_the_workstation_results_folder() -> None:
    assert grid.OUTPUT_DIR.name == "mps"
