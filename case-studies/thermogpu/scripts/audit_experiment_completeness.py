#!/usr/bin/env python3
"""Audit the frozen ThermoGPU surrogate calibration experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"

DEFAULT_CONFIG = STUDY / "configs/study-v1.toml"
DEFAULT_ROOT = STUDY / "raw-results/surrogate-performance"
DEFAULT_OUTPUT = STUDY / "processed-results/experiment-completeness.csv"


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def valid_json(path: Path) -> bool:
    if not nonempty(path):
        return False

    try:
        json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    return True


def classify(
    calibration_root: Path,
    budget: int,
    batch: int,
    dtype: str,
) -> tuple[str, str]:
    directory = calibration_root / f"budget-{budget}" / "calibration"

    success = directory / f"{dtype}-b{batch}.json"
    trials = directory / f"{dtype}-b{batch}.trials.json"
    failure = directory / f"{dtype}-b{batch}.failed.json"

    if valid_json(success):
        if valid_json(trials):
            return "complete", str(success.relative_to(ROOT))

        return (
            "complete",
            f"{success.relative_to(ROOT)} "
            "(trial detail missing or invalid)",
        )

    if valid_json(failure):
        return "failed", str(failure.relative_to(ROOT))

    if nonempty(success):
        return "failed", f"{success.relative_to(ROOT)} (invalid JSON)"

    return "pending", ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--calibration-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    with args.config.open("rb") as f:
        cfg = tomllib.load(f)

    budgets = [int(x) for x in cfg["surrogate"]["budgets"]]
    batches = [int(x) for x in cfg["benchmark"]["batch_sizes"]]

    rows = []

    for budget in budgets:
        for batch in batches:
            status, evidence = classify(
                args.calibration_root,
                budget,
                batch,
                args.dtype,
            )

            rows.append(
                {
                    "budget": budget,
                    "batch_size": batch,
                    "dtype": args.dtype,
                    "status": status,
                    "evidence": evidence,
                }
            )

    fields = [
        "budget",
        "batch_size",
        "dtype",
        "status",
        "evidence",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        status: sum(r["status"] == status for r in rows)
        for status in ("complete", "failed", "infeasible", "pending")
    }

    print(args.output)
    print(
        f"total={len(rows)} "
        f"complete={counts['complete']} "
        f"failed={counts['failed']} "
        f"infeasible={counts['infeasible']} "
        f"pending={counts['pending']}"
    )

    print()
    print("EXPERIMENT MATRIX")
    print()

    width = max(len(str(b)) for b in batches)

    print(
        f"{'budget':>6}  "
        + "  ".join(f"{b:>{width}}" for b in batches)
    )

    symbols = {
        "complete": "C",
        "failed": "F",
        "infeasible": "I",
        "pending": ".",
    }

    for budget in budgets:
        cells = []

        for batch in batches:
            row = next(
                r for r in rows
                if r["budget"] == budget
                and r["batch_size"] == batch
            )

            cells.append(
                f"{symbols[row['status']]:>{width}}"
            )

        print(f"{budget:6d}  " + "  ".join(cells))

    print()
    print("C=complete F=failed I=infeasible .=pending")


if __name__ == "__main__":
    main()
