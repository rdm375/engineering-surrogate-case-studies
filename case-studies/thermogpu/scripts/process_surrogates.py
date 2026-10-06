#!/usr/bin/env python3
"""Reduce canonical TDAR methane-Z evidence to a reporting table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    ROOT
    / "case-studies/thermogpu/raw-results/surrogates/normalized.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "case-studies/thermogpu/processed-results/surrogate-accuracy.csv"
)

FIELDS = [
    "budget",
    "points",
    "simplices",
    "tdar_seconds",
    "rmse",
    "p99_abs",
    "linf",
    "pareto_accuracy_complexity",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    data = json.loads(args.input.read_text())

    if data.get("status") != "ok":
        raise SystemExit("surrogate collection is not successful")

    rows = data["measurements"]
    if not rows:
        raise SystemExit("surrogate collection contains no measurements")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})

    print(args.output)


if __name__ == "__main__":
    main()
