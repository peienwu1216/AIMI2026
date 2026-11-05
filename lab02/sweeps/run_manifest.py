#!/usr/bin/env python3
"""Run an immutable JSONL sweep manifest through the Lab02 training runner."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _validate_unique_names(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [row["run_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate run_name")
    return rows


def expand_matrix(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    common = document.get("base", {})
    rows = []
    for group in document["groups"]:
        group_base = {**common, **group.get("base", {})}
        seeds = group.get("seeds", document.get("seeds", []))
        for variant, seed in itertools.product(group["variants"], seeds):
            label = variant["label"]
            overrides = {key: value for key, value in variant.items() if key != "label"}
            rows.append(
                {
                    **group_base,
                    **overrides,
                    "seed": seed,
                    "run_name": f"{group['name']}-{label}-s{seed}",
                }
            )
    return _validate_unique_names(path, rows)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        return expand_matrix(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        if "run_name" not in row:
            raise ValueError(f"{path}:{line_number}: missing run_name")
        rows.append(row)
    return _validate_unique_names(path, rows)


def build_command(
    row: dict[str, Any], output_dir: Path, device: str
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_DIR / "main.py"),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--run-name",
        str(row["run_name"]),
    ]
    for key, value in row.items():
        if key == "run_name" or value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "results" / "raw"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recover-incomplete",
        action="store_true",
        help="move a run directory without summary.json to results/incomplete before rerunning",
    )
    parser.add_argument("--start-index", type=int, default=1, help="1-based inclusive")
    parser.add_argument("--end-index", type=int, help="1-based inclusive")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    all_rows = read_manifest(args.manifest)
    end_index = args.end_index or len(all_rows)
    if not 1 <= args.start_index <= end_index <= len(all_rows):
        parser.error("require 1 <= start-index <= end-index <= manifest length")
    rows = all_rows[args.start_index - 1:end_index]
    print(
        f"manifest={args.manifest.resolve()} range={args.start_index}-{end_index} "
        f"runs={len(rows)} output={output_dir}"
    )
    for index, row in enumerate(rows, 1):
        run_name = row["run_name"]
        run_dir = output_dir / run_name
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            print(f"[{index}/{len(rows)}] skip complete {run_name}")
            continue
        if run_dir.exists():
            if not args.recover_incomplete:
                raise RuntimeError(
                    f"Incomplete run directory exists: {run_dir}; inspect it and rerun "
                    "with --recover-incomplete only after confirming no process owns it"
                )
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            recovery_dir = output_dir.parent / "incomplete" / timestamp / run_name
            recovery_dir.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(recovery_dir)
            print(f"[{index}/{len(rows)}] preserved incomplete {run_name} -> {recovery_dir}")
        command = build_command(row, output_dir, args.device)
        print(f"[{index}/{len(rows)}] start {run_name}", flush=True)
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        if not summary_path.exists():
            raise RuntimeError(f"Run completed without summary: {run_name}")
        print(f"[{index}/{len(rows)}] complete {run_name}", flush=True)


if __name__ == "__main__":
    main()
