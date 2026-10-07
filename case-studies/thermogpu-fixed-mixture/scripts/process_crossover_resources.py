#!/usr/bin/env python3
"""Derive surrogate crossover and hardware-resource tables for Case Study 2."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

DEFAULT_COMBINED = STUDY / "processed-results/combined-candidates.csv"
DEFAULT_SURROGATES = STUDY / "processed-results/surrogate-performance.csv"

DEFAULT_CROSSOVER = STUDY / "processed-results/surrogate-crossover.csv"
DEFAULT_RESOURCES = STUDY / "processed-results/resource-pressure.csv"

COMPONENTS = (1, 2, 3, 4, 5)
BATCHES = (1, 10, 100, 1000, 10000, 100000, 1000000)
BUDGETS = (16, 24, 32, 48, 64, 96, 128)

CROSSOVER_FIELDS = [
    "components",
    "batch",
    "best_exact_method",
    "best_exact_backend",
    "best_exact_ns_per_eval",
    "any_surrogate_faster",
    "faster_surrogate_count",
    "most_accurate_faster_budget",
    "most_accurate_faster_rmse",
    "most_accurate_faster_ns_per_eval",
    "most_accurate_faster_speedup",
    "fastest_surrogate_budget",
    "fastest_surrogate_rmse",
    "fastest_surrogate_ns_per_eval",
    "max_speedup",
]

RESOURCE_FIELDS = [
    "components",
    "budget",
    "rmse",
    "p99_abs",
    "max_abs",
    "points",
    "simplices",
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
    "has_spills",
    "previous_budget",
    "delta_rmse",
    "rmse_ratio_vs_previous",
    "delta_dag_nodes",
    "delta_executed_ops",
    "delta_live_slots",
    "delta_registers",
    "delta_spill_store_bytes",
    "delta_spill_load_bytes",
    "delta_stack_frame_bytes",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
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


def process_crossover(raw: list[dict[str, str]]) -> list[dict]:
    if len(raw) != 490:
        raise ValueError(f"expected 490 combined rows, found {len(raw)}")

    groups = defaultdict(list)
    for r in raw:
        groups[(int(r["components"]), int(r["batch"]))].append(r)

    expected = {(c, b) for c in COMPONENTS for b in BATCHES}
    if set(groups) != expected:
        raise ValueError("incomplete crossover operating-point coverage")

    result = []

    for c, batch in sorted(groups):
        group = groups[(c, batch)]
        exact = [r for r in group if r["candidate_type"] == "exact"]
        sur = [r for r in group if r["candidate_type"] == "surrogate"]

        if len(exact) != 7 or len(sur) != 7:
            raise ValueError(
                f"C{c} batch {batch}: expected 7 exact + 7 surrogate rows"
            )

        best_exact = min(exact, key=lambda r: float(r["ns_per_eval"]))
        exact_ns = float(best_exact["ns_per_eval"])

        faster = [
            r for r in sur
            if float(r["ns_per_eval"]) < exact_ns
        ]

        fastest = min(sur, key=lambda r: float(r["ns_per_eval"]))

        # Among measured surrogates that beat exact, choose the one with
        # smallest measured RMSE. No interpolation between budgets.
        most_accurate = (
            min(faster, key=lambda r: float(r["rmse"]))
            if faster
            else None
        )

        result.append(
            {
                "components": c,
                "batch": batch,
                "best_exact_method": best_exact["method"],
                "best_exact_backend": best_exact["backend"],
                "best_exact_ns_per_eval": exact_ns,
                "any_surrogate_faster": bool(faster),
                "faster_surrogate_count": len(faster),
                "most_accurate_faster_budget":
                    int(most_accurate["budget"]) if most_accurate else "",
                "most_accurate_faster_rmse":
                    float(most_accurate["rmse"]) if most_accurate else "",
                "most_accurate_faster_ns_per_eval":
                    float(most_accurate["ns_per_eval"])
                    if most_accurate else "",
                "most_accurate_faster_speedup":
                    exact_ns / float(most_accurate["ns_per_eval"])
                    if most_accurate else "",
                "fastest_surrogate_budget": int(fastest["budget"]),
                "fastest_surrogate_rmse": float(fastest["rmse"]),
                "fastest_surrogate_ns_per_eval":
                    float(fastest["ns_per_eval"]),
                "max_speedup":
                    exact_ns / float(fastest["ns_per_eval"]),
            }
        )

    return result


def invariant_value(group: list[dict[str, str]], field: str):
    values = {r[field] for r in group}
    if len(values) != 1:
        raise ValueError(
            f"{field} varies across batches: {sorted(values)!r}"
        )
    return next(iter(values))


def process_resources(raw: list[dict[str, str]]) -> list[dict]:
    if len(raw) != 245:
        raise ValueError(f"expected 245 surrogate rows, found {len(raw)}")

    groups = defaultdict(list)
    for r in raw:
        groups[(int(r["components"]), int(r["budget"]))].append(r)

    expected = {(c, b) for c in COMPONENTS for b in BUDGETS}
    if set(groups) != expected:
        raise ValueError("incomplete resource artifact coverage")

    invariant_fields = [
        "rmse",
        "p99_abs",
        "max_abs",
        "points",
        "simplices",
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
    ]

    rows = []

    for c in COMPONENTS:
        previous = None

        for budget in BUDGETS:
            group = groups[(c, budget)]

            if len(group) != 7:
                raise ValueError(
                    f"C{c} B{budget}: expected 7 batch measurements"
                )

            values = {
                field: invariant_value(group, field)
                for field in invariant_fields
            }

            current = {
                "components": c,
                "budget": budget,
                "rmse": float(values["rmse"]),
                "p99_abs": float(values["p99_abs"]),
                "max_abs": float(values["max_abs"]),
                "points": int(values["points"]),
                "simplices": int(values["simplices"]),
                "dag_nodes": int(values["dag_nodes"]),
                "affine_nodes": int(values["affine_nodes"]),
                "min_nodes": int(values["min_nodes"]),
                "max_nodes": int(values["max_nodes"]),
                "executed_ops": int(values["executed_ops"]),
                "live_slots": int(values["live_slots"]),
                "registers": int(values["registers"]),
                "spill_store_bytes": int(values["spill_store_bytes"]),
                "spill_load_bytes": int(values["spill_load_bytes"]),
                "stack_frame_bytes": int(values["stack_frame_bytes"]),
                "has_spills":
                    int(values["spill_store_bytes"]) > 0
                    or int(values["spill_load_bytes"]) > 0,
                "previous_budget": "",
                "delta_rmse": "",
                "rmse_ratio_vs_previous": "",
                "delta_dag_nodes": "",
                "delta_executed_ops": "",
                "delta_live_slots": "",
                "delta_registers": "",
                "delta_spill_store_bytes": "",
                "delta_spill_load_bytes": "",
                "delta_stack_frame_bytes": "",
            }

            if previous is not None:
                current["previous_budget"] = previous["budget"]
                current["delta_rmse"] = (
                    current["rmse"] - previous["rmse"]
                )
                current["rmse_ratio_vs_previous"] = (
                    current["rmse"] / previous["rmse"]
                )

                for field in [
                    "dag_nodes",
                    "executed_ops",
                    "live_slots",
                    "registers",
                    "spill_store_bytes",
                    "spill_load_bytes",
                    "stack_frame_bytes",
                ]:
                    current[f"delta_{field}"] = (
                        current[field] - previous[field]
                    )

            rows.append(current)
            previous = current

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    p.add_argument("--surrogates", type=Path, default=DEFAULT_SURROGATES)
    p.add_argument("--crossover-output", type=Path, default=DEFAULT_CROSSOVER)
    p.add_argument("--resources-output", type=Path, default=DEFAULT_RESOURCES)
    a = p.parse_args()

    crossover = process_crossover(read(a.combined))
    resources = process_resources(read(a.surrogates))

    write(a.crossover_output, crossover, CROSSOVER_FIELDS)
    write(a.resources_output, resources, RESOURCE_FIELDS)

    print(f"crossover rows={len(crossover)}")
    print(f"resource rows={len(resources)}")


if __name__ == "__main__":
    main()
