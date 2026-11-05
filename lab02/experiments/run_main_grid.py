"""Run the final recipe and the P3 and P4 groups through ``main.py`` on the local GPU.

    python project/experiments/run_main_grid.py --groups p4,p3 --workers 2

The workstation sweep starts from P0 and this script starts from the end of the
list, so the two meet in the middle. Output goes to ``results/mps/`` and never to
``results/raw/``, which belongs to the workstation sweep. Finished runs are
skipped, so a batch can be started again after an interruption.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "results" / "mps"
SEEDS = (17, 42, 113)
FINAL_SEEDS = (17, 42, 113, 511, 2026)  # same seeds as the workstation P0 runs
MODES = {
    "full": ["--val-ratio", "0", "--select-by", "test_accuracy"],
    "val": ["--val-ratio", "0.2", "--select-by", "val_accuracy"],
}
TUNED_EEGNET = ["--model", "eegnet", "--activation", "leaky_relu", "--dropout", "0.5", "--lr", "1e-3"]


def bonus_settings():
    settings = {}
    for lr in ("1e-2", "1e-3", "3e-4"):
        settings[f"deep-elu-lr{lr}"] = [
            "--model", "deepconvnet", "--activation", "elu", "--dropout", "0.5", "--lr", lr,
        ]
        settings[f"shallow-lr{lr}"] = ["--model", "shallowconvnet", "--dropout", "0.5", "--lr", lr]
    for activation in ("relu", "leaky_relu"):
        settings[f"deep-{activation}-lr1e-3"] = [
            "--model", "deepconvnet", "--activation", activation, "--dropout", "0.5", "--lr", "1e-3",
        ]
    return settings


def architecture_settings():
    settings = {"eegnet-tuned": TUNED_EEGNET}
    for length in (15, 25, 33, 75, 101):
        settings[f"kernel{length}"] = TUNED_EEGNET + ["--kernel-length", str(length)]
    for length in (7, 31):
        settings[f"sepkernel{length}"] = TUNED_EEGNET + ["--sep-kernel", str(length)]
    settings["true-separable"] = TUNED_EEGNET + ["--separable"]
    for f1, depth in ((8, 2), (32, 2), (16, 4)):
        settings[f"f{f1}d{depth}"] = TUNED_EEGNET + [
            "--f1", str(f1), "--depth-multiplier", str(depth), "--f2", str(f1 * depth),
        ]
    return settings


GROUPS = {"p4": bonus_settings, "p3": architecture_settings}


def final_jobs():
    """Final recipe: 300 epochs on all 1,080 trials, paired with the P0 seeds."""
    jobs = []
    for activation in ("relu", "leaky_relu"):
        for seed in FINAL_SEEDS:
            name = f"final300-{activation}-s{seed}"
            jobs.append(
                (
                    name,
                    ["--model", "eegnet", "--activation", activation, "--dropout", "0.5", "--lr", "1e-3"]
                    + ["--num-epochs", "300"]
                    + MODES["full"]
                    + ["--seed", str(seed), "--device", "mps", "--run-name", name],
                )
            )
    return jobs


def build_jobs(group_names, labels=None):
    jobs = []
    for group in group_names:
        if group == "final300":
            jobs.extend(final_jobs())
            continue
        for label, settings in GROUPS[group]().items():
            if labels is not None and label not in labels:
                continue
            for mode, mode_args in MODES.items():
                for seed in SEEDS:
                    name = f"{group}-{label}-{mode}-s{seed}"
                    jobs.append(
                        (
                            name,
                            settings
                            + mode_args
                            + ["--seed", str(seed), "--device", "mps", "--run-name", name],
                        )
                    )
    return jobs


def run_job(job):
    name, args = job
    if (OUTPUT_DIR / name / "summary.json").exists():
        return name, "skipped"
    if (OUTPUT_DIR / name).exists():
        # main.py creates the folder first, so another launcher owns this run.
        return name, "in progress elsewhere"
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MPLCONFIGDIR": "/tmp/aimi_lab02_mpl"}
    with (log_dir / f"{name}.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "main.py"), *args, "--output-dir", str(OUTPUT_DIR)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    return name, "done" if result.returncode == 0 else f"FAILED ({result.returncode})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="p4,p3", help="comma-separated: final300, p4, p3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--reverse", action="store_true", help="start from the end, to share a batch with a second launcher"
    )
    parser.add_argument(
        "--labels", help="comma-separated settings to keep, when another machine owns the rest"
    )
    args = parser.parse_args()

    labels = set(args.labels.split(",")) if args.labels else None
    jobs = build_jobs(args.groups.split(","), labels)
    if args.reverse:
        jobs.reverse()
    print(f"{len(jobs)} runs", flush=True)
    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            name, status = future.result()
            print(f"[{done}/{len(jobs)}] {name}: {status}", flush=True)
            if status.startswith("FAILED"):
                failures.append(name)
    print(f"finished, {len(failures)} failed: {failures}", flush=True)


if __name__ == "__main__":
    main()
