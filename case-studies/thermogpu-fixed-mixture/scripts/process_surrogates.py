#!/usr/bin/env python3
"""Normalize retained fixed-mixture TDAR/CPWA hardware evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

DEFAULT_INPUT = STUDY / "raw-results/surrogates/summary.csv"
DEFAULT_OUTPUT = STUDY / "processed-results/surrogate-performance.csv"

COMPONENTS = {1, 2, 3, 4, 5}
BUDGETS = {16, 24, 32, 48, 64, 96, 128}
BATCHES = {1, 10, 100, 1000, 10000, 100000, 1000000}

FIELDS = [
    "components",
    "budget",
    "batch",
    "rmse",
    "p99_abs",
    "max_abs",
    "points",
    "simplices",
    "tdar_seconds",
    "ns_per_eval",
    "evaluations_per_second",
    "dag_nodes",
    "affine_nodes",
    "min_nodes",
    "max_nodes",
    "executed_ops",
    "live_slots",
    "registers",
    "spill_store_bytes",
    "spill_load_bytes",
    "stack_frame_bytes",
    "max_delta_cpu",
    "tdar_artifact",
    "cpwa_artifact",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def process(raw: list[dict[str, str]]):
    if len(raw) != 245:
        raise ValueError(f"expected 245 surrogate rows, found {len(raw)}")

    rows = []

    for r in raw:
        components = int(r["components"])
        budget = int(r["budget"])
        batch = int(r["batch"])

        if components not in COMPONENTS:
            raise ValueError(f"unexpected component count {components}")
        if budget not in BUDGETS:
            raise ValueError(f"unexpected budget {budget}")
        if batch not in BATCHES:
            raise ValueError(f"unexpected batch {batch}")

        floats = {
            "rmse": float(r["rmse"]),
            "p99_abs": float(r["p99_abs"]),
            "max_abs": float(r["max_abs"]),
            "tdar_seconds": float(r["tdar_seconds"]),
            "ns_per_eval": float(r["ns_per_eval"]),
            "max_delta_cpu": float(r["max_delta_cpu"]),
        }

        for name, value in floats.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"invalid {name} for C{components} B{budget} "
                    f"batch {batch}: {value}"
                )

        if floats["max_delta_cpu"] != 0.0:
            raise ValueError(
                f"CPU/CUDA mismatch for C{components} B{budget} "
                f"batch {batch}: {floats['max_delta_cpu']}"
            )

        ns = floats["ns_per_eval"]

        rows.append(
            {
                "components": components,
                "budget": budget,
                "batch": batch,
                "rmse": floats["rmse"],
                "p99_abs": floats["p99_abs"],
                "max_abs": floats["max_abs"],
                "points": int(r["points"]),
                "simplices": int(r["simplices"]),
                "tdar_seconds": floats["tdar_seconds"],
                "ns_per_eval": ns,
                "evaluations_per_second": 1e9 / ns,
                "dag_nodes": int(r["dag_nodes"]),
                "affine_nodes": int(r["affine_nodes"]),
                "min_nodes": int(r["min_nodes"]),
                "max_nodes": int(r["max_nodes"]),
                "executed_ops": int(r["executed_ops"]),
                "live_slots": int(r["live_slots"]),
                "registers": int(r["registers"]),
                "spill_store_bytes": int(r["spill_store_bytes"]),
                "spill_load_bytes": int(r["spill_load_bytes"]),
                "stack_frame_bytes": int(r["stack_frame_bytes"]),
                "max_delta_cpu": floats["max_delta_cpu"],
                "tdar_artifact": r["tdar_artifact"],
                "cpwa_artifact": r["cpwa_artifact"],
            }
        )

    groups = {}
    for r in rows:
        groups.setdefault(
            (r["components"], r["budget"]),
            []
        ).append(r)

    expected_groups = {
        (components, budget)
        for components in COMPONENTS
        for budget in BUDGETS
    }
    if set(groups) != expected_groups:
        raise ValueError("surrogate artifact coverage is incomplete")

    for key, group in groups.items():
        if len(group) != 7:
            raise ValueError(
                f"expected 7 batches for {key}, found {len(group)}"
            )

        if {r["batch"] for r in group} != BATCHES:
            raise ValueError(f"incomplete batch coverage for {key}")

        # Accuracy and DAG/resource structure are properties of one
        # compiled approximation and must be invariant across batch size.
        invariant_fields = [
            "rmse",
            "p99_abs",
            "max_abs",
            "points",
            "simplices",
            "tdar_seconds",
            "dag_nodes",
            "affine_nodes",
            "min_nodes",
            "max_nodes",
            "executed_ops",
            "live_slots",
            "registers",
            "spill_store_bytes",
            "spill_load_bytes",
            "stack_frame_bytes",
            "max_delta_cpu",
            "tdar_artifact",
            "cpwa_artifact",
        ]

        for field in invariant_fields:
            values = {r[field] for r in group}
            if len(values) != 1:
                raise ValueError(
                    f"{key}: {field} varies across batch: {values}"
                )

    rows.sort(
        key=lambda r: (
            r["components"],
            r["budget"],
            r["batch"],
        )
    )

    return rows


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()

    rows = process(read(a.input))
    write(a.output, rows)

    print(f"surrogate rows={len(rows)}")
    print(
        "surrogate artifacts="
        f"{len({(r['components'], r['budget']) for r in rows})}"
    )


if __name__ == "__main__":
    main()
