import json
from pathlib import Path

import pytest

from sweeps.run_manifest import build_command, read_manifest


def test_p0_manifest_has_unique_expected_runs() -> None:
    rows = read_manifest(Path(__file__).parents[1] / "sweeps" / "p0_final.jsonl")
    assert len(rows) == 10
    assert {row["activation"] for row in rows} == {"relu", "leaky_relu"}
    assert {row["seed"] for row in rows} == {17, 42, 113, 511, 2026}
    assert all(row["val_ratio"] == 0 for row in rows)
    assert all(row["select_by"] == "test_accuracy" for row in rows)


def test_p1_matrix_expands_to_99_unique_comparison_runs() -> None:
    rows = read_manifest(Path(__file__).parents[1] / "sweeps" / "p1_core_matrix.json")
    assert len(rows) == 99
    assert len({row["run_name"] for row in rows}) == 99
    assert {row["seed"] for row in rows} == {17, 42, 113}
    assert all(row["val_ratio"] == 0.2 for row in rows)
    assert all(row["select_by"] == "val_accuracy" for row in rows)


def test_bonus_matrix_covers_two_models_three_lrs_three_seeds() -> None:
    rows = read_manifest(Path(__file__).parents[1] / "sweeps" / "p4_bonus_matrix.json")
    assert len(rows) == 18
    assert {row["model"] for row in rows} == {"deepconvnet", "shallowconvnet"}
    assert {row["lr"] for row in rows} == {0.01, 0.001, 0.0003}
    assert {row["seed"] for row in rows} == {17, 42, 113}


def test_p2_matrix_has_57_optimization_runs() -> None:
    rows = read_manifest(
        Path(__file__).parents[1] / "sweeps" / "p2_optimization_matrix.json"
    )
    assert len(rows) == 57
    assert len({row["run_name"] for row in rows}) == 57
    assert {row["seed"] for row in rows} == {17, 42, 113}


def test_build_command_converts_values_and_boolean_flags(tmp_path) -> None:
    command = build_command(
        {"run_name": "x", "model": "eegnet", "separable": True, "dropout": 0.5},
        tmp_path,
        "cuda",
    )
    assert "--run-name" in command
    assert "--separable" in command
    assert command[command.index("--dropout") + 1] == "0.5"


def test_manifest_rejects_duplicate_names(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps({"run_name": "same"}) + "\n" + json.dumps({"run_name": "same"})
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)
