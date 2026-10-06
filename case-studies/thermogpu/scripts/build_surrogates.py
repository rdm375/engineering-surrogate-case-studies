#!/usr/bin/env python3
"""Collect the canonical frozen TDAR methane-Z surrogate sweep.

The released TDAR ThermoGPU tradeoff driver owns surrogate construction and
held-out validation.  This collector invokes that driver once for all requested
budgets so every surrogate is evaluated against the same frozen validation
design and ThermoGPU truth vector.

Native TDAR outputs are retained verbatim.  This script adds case-study
provenance and execution metadata without reimplementing TDAR's experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from engineering_case_studies.collection import (
    run_capture,
    slurm_metadata,
    write_json,
)
from engineering_case_studies.provenance import git_state


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "case-studies/thermogpu/raw-results/surrogates"
STUDY = "thermogpu-methane-z-v1"

DEFAULT_BUDGETS = [16, 32, 64, 128, 256]


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    p.add_argument(
        "--initial",
        type=int,
        default=16,
    )
    p.add_argument(
        "--tdar-root",
        type=Path,
        default=Path("/nvme/Sync/tdar"),
    )
    p.add_argument(
        "--thermogpu-root",
        type=Path,
        default=Path("/nvme/Sync/ThermoGPU"),
    )
    p.add_argument(
        "--validation-points",
        type=int,
        default=100000,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20261002,
    )

    args = p.parse_args()

    budgets = sorted(set(args.budgets))

    oracle = (
        args.thermogpu_root
        / "build"
        / "thermogpu_tdar_oracle"
    )
    driver = (
        args.tdar_root
        / "benchmarks"
        / "run_thermogpu_tradeoff.py"
    )

    if not oracle.exists():
        raise SystemExit(f"ThermoGPU TDAR oracle not found: {oracle}")

    if not driver.exists():
        raise SystemExit(f"TDAR tradeoff driver not found: {driver}")

    RAW.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(driver),
        "--oracle",
        str(oracle),
        "--budgets",
        *map(str, budgets),
        "--initial",
        str(args.initial),
        "--validation-points",
        str(args.validation_points),
        "--seed",
        str(args.seed),
        "--output",
        str(RAW),
    ]

    env = dict(os.environ)

    tdar_src = str(args.tdar_root / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = tdar_src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = tdar_src

    run = run_capture(
        cmd,
        cwd=args.tdar_root,
        env=env,
    )

    (RAW / "collector.stdout.txt").write_text(run["stdout"])
    (RAW / "collector.stderr.txt").write_text(run["stderr"])

    if run["returncode"] != 0:
        write_json(
            RAW / "collector.json",
            {
                "status": "failed",
                **run,
                "slurm": slurm_metadata(),
            },
        )
        raise SystemExit("TDAR tradeoff sweep failed")

    summary_path = RAW / "summary.csv"
    report_path = RAW / "report.json"
    validation_points_path = RAW / "validation-points.npy"
    validation_truth_path = RAW / "validation-truth.npy"

    required = [
        summary_path,
        report_path,
        validation_points_path,
        validation_truth_path,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise SystemExit(
            "TDAR sweep completed but required outputs are missing:\n"
            + "\n".join(missing)
        )

    rows = read_summary(summary_path)

    observed_budgets = sorted(
        int(row["budget"])
        for row in rows
    )

    if observed_budgets != budgets:
        raise SystemExit(
            f"budget mismatch: requested={budgets}, "
            f"observed={observed_budgets}"
        )

    normalized_rows = []

    for row in rows:
        normalized_rows.append(
            {
                "budget": int(row["budget"]),
                "points": int(row["points"]),
                "simplices": int(row["simplices"]),
                "tdar_seconds": float(row["tdar_seconds"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "p95_abs": float(row["p95_abs"]),
                "p99_abs": float(row["p99_abs"]),
                "linf": float(row["max_abs"]),
                "pareto_accuracy_complexity":
                    row["pareto_accuracy_complexity"].lower()
                    in {"true", "1", "yes"},
                "artifact": row["artifact"],
            }
        )

    record = {
        "status": "ok",
        "study": STUDY,
        "kind": "surrogate_accuracy_sweep",
        "method": "tdar-simplicial-cpwa",
        "budgets": budgets,
        "initial_points": args.initial,
        "validation": {
            "method": "sobol",
            "points": args.validation_points,
            "seed": args.seed + 1,
            "points_file": str(
                validation_points_path.relative_to(ROOT)
            ),
            "truth_file": str(
                validation_truth_path.relative_to(ROOT)
            ),
        },
        "measurements": normalized_rows,
        "wall_seconds": run["elapsed_seconds"],
        "source": "TDAR",
        "tdar": git_state(args.tdar_root),
        "thermogpu": git_state(args.thermogpu_root),
        "slurm": slurm_metadata(),
        "command": run["command"],
    }

    write_json(RAW / "normalized.json", record)

    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
