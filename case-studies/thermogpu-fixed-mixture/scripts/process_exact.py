#!/usr/bin/env python3
"""Normalize retained exact fixed-mixture EOS performance evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

DEFAULT_INPUT = STUDY / "raw-results/exact/exact-scaling.csv"
DEFAULT_OUTPUT = STUDY / "processed-results/exact-performance.csv"
DEFAULT_CROSSOVER = STUDY / "processed-results/exact-crossover.csv"

COMPONENTS = {1, 2, 3, 4, 5}
BATCHES = {1, 10, 100, 1000, 10000, 100000, 1000000}

FIELDS = [
    "components",
    "batch",
    "method",
    "backend",
    "threads",
    "calls_per_sample",
    "median_seconds",
    "ns_per_eval",
    "evaluations_per_second",
    "mad_percent",
    "is_best_exact",
]

CROSSOVER_FIELDS = [
    "components",
    "batch",
    "method",
    "backend",
    "threads",
    "ns_per_eval",
    "evaluations_per_second",
    "mad_percent",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def method_name(backend: str, threads: int) -> str:
    if backend == "scalar":
        return "CPU scalar"
    if backend == "openmp":
        return f"CPU OpenMP {threads}"
    if backend == "cuda_generic_resident":
        return "GPU generic resident"
    if backend == "cuda_fixed_resident":
        return "GPU fixed-composition resident"
    raise ValueError(f"unknown backend {backend!r}")


def process(raw: list[dict[str, str]]):
    if len(raw) != 245:
        raise ValueError(f"expected 245 exact rows, found {len(raw)}")

    rows = []

    for r in raw:
        components = int(r["components"])
        batch = int(r["states"])
        backend = r["backend"]
        threads = int(r["threads"])

        ns = float(r["ns_per_state"])
        rate = float(r["states_per_second"])
        median_seconds = float(r["median_seconds"])
        mad = float(r["mad_percent"])

        if components not in COMPONENTS:
            raise ValueError(f"unexpected component count {components}")
        if batch not in BATCHES:
            raise ValueError(f"unexpected batch {batch}")

        for name, value in [
            ("ns_per_state", ns),
            ("states_per_second", rate),
            ("median_seconds", median_seconds),
            ("mad_percent", mad),
        ]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"invalid {name} for C{components}, batch {batch}: {value}"
                )

        rows.append(
            {
                "components": components,
                "batch": batch,
                "method": method_name(backend, threads),
                "backend": backend,
                "threads": threads,
                "calls_per_sample": int(r["calls_per_sample"]),
                "median_seconds": median_seconds,
                "ns_per_eval": ns,
                "evaluations_per_second": rate,
                "mad_percent": mad,
                "is_best_exact": False,
            }
        )

    groups = {}
    for r in rows:
        groups.setdefault((r["components"], r["batch"]), []).append(r)

    expected_groups = {
        (components, batch)
        for components in COMPONENTS
        for batch in BATCHES
    }
    if set(groups) != expected_groups:
        raise ValueError("exact operating-point coverage is incomplete")

    winners = []

    for key, group in groups.items():
        if len(group) != 7:
            raise ValueError(
                f"expected 7 exact methods for {key}, found {len(group)}"
            )

        backends = [r["backend"] for r in group]

        if backends.count("scalar") != 1:
            raise ValueError(f"{key}: expected one scalar row")
        if backends.count("openmp") != 4:
            raise ValueError(f"{key}: expected four OpenMP rows")
        if backends.count("cuda_generic_resident") != 1:
            raise ValueError(f"{key}: expected one cuda_generic row")
        if backends.count("cuda_fixed_resident") != 1:
            raise ValueError(f"{key}: expected one cuda_fixed row")

        omp_threads = {
            r["threads"]
            for r in group
            if r["backend"] == "openmp"
        }
        if omp_threads != {1, 2, 4, 8}:
            raise ValueError(
                f"{key}: unexpected OpenMP thread set {omp_threads}"
            )

        winner = min(group, key=lambda r: r["ns_per_eval"])
        winner["is_best_exact"] = True
        winners.append(winner)

    rows.sort(
        key=lambda r: (
            r["components"],
            r["batch"],
            r["ns_per_eval"],
            r["backend"],
            r["threads"],
        )
    )
    winners.sort(key=lambda r: (r["components"], r["batch"]))

    return rows, winners


def write(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--crossover-output",
        type=Path,
        default=DEFAULT_CROSSOVER,
    )
    a = p.parse_args()

    rows, winners = process(read(a.input))

    write(a.output, rows, FIELDS)
    write(a.crossover_output, winners, CROSSOVER_FIELDS)

    print(f"exact rows={len(rows)}")
    print(f"operating points={len(winners)}")


if __name__ == "__main__":
    main()
