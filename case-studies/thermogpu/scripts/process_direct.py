#!/usr/bin/env python3
"""Reduce canonical ThermoGPU scaling evidence to methane reporting data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    ROOT
    / "case-studies/thermogpu/raw-results/direct"
    / "scaling-b1-10-100-1000-10000-100000-1000000.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "case-studies/thermogpu/processed-results/direct-methane.csv"
)


FIELDS = [
    "batch",
    "scalar_ns",
    "scalar_rate",
    "best_openmp_ns",
    "best_openmp_rate",
    "best_openmp_threads",
    "cuda_resident_ns",
    "cuda_resident_rate",
    "cuda_e2e_ns",
    "cuda_e2e_rate",
    "openmp_speedup_vs_scalar",
    "cuda_resident_speedup_vs_scalar",
    "cuda_resident_speedup_vs_best_cpu",
    "cuda_e2e_speedup_vs_scalar",
    "cuda_e2e_speedup_vs_best_cpu",
]


def one(rows, backend):
    matches = [r for r in rows if r["backend"] == backend]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {backend!r} row, found {len(matches)}"
        )
    return matches[0]


def process(record):
    if record.get("status") != "ok":
        raise ValueError("input record is not a successful collection")

    methane = [
        r for r in record["measurements"]
        if int(r["components"]) == 1
    ]

    batches = sorted({int(r["states"]) for r in methane})
    result = []

    for batch in batches:
        rows = [r for r in methane if int(r["states"]) == batch]

        scalar = one(rows, "scalar")
        resident = one(rows, "cuda_resident")
        e2e = one(rows, "cuda_e2e")

        openmp_rows = [r for r in rows if r["backend"] == "openmp"]
        if not openmp_rows:
            raise ValueError(f"no OpenMP measurements for batch {batch}")

        best_openmp = min(
            openmp_rows,
            key=lambda r: float(r["ns_per_state"]),
        )

        scalar_ns = float(scalar["ns_per_state"])
        openmp_ns = float(best_openmp["ns_per_state"])
        resident_ns = float(resident["ns_per_state"])
        e2e_ns = float(e2e["ns_per_state"])

        result.append(
            {
                "batch": batch,
                "scalar_ns": scalar_ns,
                "scalar_rate": float(scalar["states_per_second"]),
                "best_openmp_ns": openmp_ns,
                "best_openmp_rate": float(
                    best_openmp["states_per_second"]
                ),
                "best_openmp_threads": int(best_openmp["threads"]),
                "cuda_resident_ns": resident_ns,
                "cuda_resident_rate": float(
                    resident["states_per_second"]
                ),
                "cuda_e2e_ns": e2e_ns,
                "cuda_e2e_rate": float(e2e["states_per_second"]),
                "openmp_speedup_vs_scalar": scalar_ns / openmp_ns,
                "cuda_resident_speedup_vs_scalar":
                    scalar_ns / resident_ns,
                "cuda_resident_speedup_vs_best_cpu":
                    openmp_ns / resident_ns,
                "cuda_e2e_speedup_vs_scalar":
                    scalar_ns / e2e_ns,
                "cuda_e2e_speedup_vs_best_cpu":
                    openmp_ns / e2e_ns,
            }
        )

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    record = json.loads(args.input.read_text())
    rows = process(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)


if __name__ == "__main__":
    main()
