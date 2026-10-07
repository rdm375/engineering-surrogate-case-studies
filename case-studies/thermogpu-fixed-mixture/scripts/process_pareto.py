#!/usr/bin/env python3
"""Construct batch-conditioned Pareto frontiers for Case Study 2."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

DEFAULT_EXACT = STUDY / "processed-results/exact-performance.csv"
DEFAULT_SURROGATE = STUDY / "processed-results/surrogate-performance.csv"
DEFAULT_COMBINED = STUDY / "processed-results/combined-candidates.csv"
DEFAULT_PARETO = STUDY / "processed-results/pareto-frontiers.csv"

COMPONENTS = {1, 2, 3, 4, 5}
BATCHES = {1, 10, 100, 1000, 10000, 100000, 1000000}
BUDGETS = {16, 24, 32, 48, 64, 96, 128}

FIELDS = [
    "components",
    "batch",
    "candidate_type",
    "method",
    "backend",
    "budget",
    "threads",
    "rmse",
    "p99_abs",
    "max_abs",
    "ns_per_eval",
    "evaluations_per_second",
    "is_best_exact",
    "best_exact_ns_per_eval",
    "speedup_vs_best_exact",
    "cost_ratio_vs_best_exact",
    "pareto",
    "surrogate_pareto",
    "registers",
    "spill_store_bytes",
    "spill_load_bytes",
    "stack_frame_bytes",
    "live_slots",
    "executed_ops",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def nondominated(rows: list[dict]) -> set[int]:
    """Return object IDs of points not strictly Pareto dominated."""
    keep = set()

    for r in rows:
        error = float(r["rmse"])
        cost = float(r["ns_per_eval"])

        dominated = any(
            (
                float(q["rmse"]) <= error
                and float(q["ns_per_eval"]) <= cost
            )
            and (
                float(q["rmse"]) < error
                or float(q["ns_per_eval"]) < cost
            )
            for q in rows
            if q is not r
        )

        if not dominated:
            keep.add(id(r))

    return keep


def exact_candidates(raw: list[dict[str, str]]) -> list[dict]:
    if len(raw) != 245:
        raise ValueError(f"expected 245 exact rows, found {len(raw)}")

    rows = []

    for r in raw:
        rows.append(
            {
                "components": int(r["components"]),
                "batch": int(r["batch"]),
                "candidate_type": "exact",
                "method": r["method"],
                "backend": r["backend"],
                "budget": "",
                "threads": int(r["threads"]),
                "rmse": 0.0,
                "p99_abs": 0.0,
                "max_abs": 0.0,
                "ns_per_eval": float(r["ns_per_eval"]),
                "evaluations_per_second":
                    float(r["evaluations_per_second"]),
                "is_best_exact": r["is_best_exact"] == "True",
                "best_exact_ns_per_eval": "",
                "speedup_vs_best_exact": "",
                "cost_ratio_vs_best_exact": "",
                "pareto": False,
                "surrogate_pareto": False,
                "registers": "",
                "spill_store_bytes": "",
                "spill_load_bytes": "",
                "stack_frame_bytes": "",
                "live_slots": "",
                "executed_ops": "",
            }
        )

    return rows


def surrogate_candidates(raw: list[dict[str, str]]) -> list[dict]:
    if len(raw) != 245:
        raise ValueError(f"expected 245 surrogate rows, found {len(raw)}")

    rows = []

    for r in raw:
        budget = int(r["budget"])

        rows.append(
            {
                "components": int(r["components"]),
                "batch": int(r["batch"]),
                "candidate_type": "surrogate",
                "method": f"GPU surrogate B{budget}",
                "backend": "gpu_surrogate",
                "budget": budget,
                "threads": 0,
                "rmse": float(r["rmse"]),
                "p99_abs": float(r["p99_abs"]),
                "max_abs": float(r["max_abs"]),
                "ns_per_eval": float(r["ns_per_eval"]),
                "evaluations_per_second":
                    float(r["evaluations_per_second"]),
                "is_best_exact": False,
                "best_exact_ns_per_eval": "",
                "speedup_vs_best_exact": "",
                "cost_ratio_vs_best_exact": "",
                "pareto": False,
                "surrogate_pareto": False,
                "registers": int(r["registers"]),
                "spill_store_bytes": int(r["spill_store_bytes"]),
                "spill_load_bytes": int(r["spill_load_bytes"]),
                "stack_frame_bytes": int(r["stack_frame_bytes"]),
                "live_slots": int(r["live_slots"]),
                "executed_ops": int(r["executed_ops"]),
            }
        )

    return rows


def annotate(rows: list[dict]) -> None:
    groups = defaultdict(list)

    for r in rows:
        groups[(r["components"], r["batch"])].append(r)

    expected = {
        (components, batch)
        for components in COMPONENTS
        for batch in BATCHES
    }

    if set(groups) != expected:
        raise ValueError("incomplete (components, batch) coverage")

    for key, group in groups.items():
        exact = [r for r in group if r["candidate_type"] == "exact"]
        surrogates = [
            r for r in group
            if r["candidate_type"] == "surrogate"
        ]

        if len(exact) != 7:
            raise ValueError(f"{key}: expected 7 exact candidates")
        if len(surrogates) != 7:
            raise ValueError(f"{key}: expected 7 surrogate candidates")

        if {int(r["budget"]) for r in surrogates} != BUDGETS:
            raise ValueError(f"{key}: incomplete surrogate budget set")

        if any(float(r["rmse"]) != 0.0 for r in exact):
            raise ValueError(f"{key}: exact candidate has nonzero RMSE")

        best_exact = min(
            exact,
            key=lambda r: float(r["ns_per_eval"]),
        )
        best_exact_ns = float(best_exact["ns_per_eval"])

        annotated_best = [r for r in exact if r["is_best_exact"]]
        if len(annotated_best) != 1:
            raise ValueError(
                f"{key}: expected one pre-annotated best exact candidate"
            )
        if annotated_best[0] is not best_exact:
            raise ValueError(
                f"{key}: best-exact annotation disagrees with timing"
            )

        all_keep = nondominated(group)
        surrogate_keep = nondominated(surrogates)

        for r in group:
            r["best_exact_ns_per_eval"] = best_exact_ns
            r["pareto"] = id(r) in all_keep

            if r["candidate_type"] == "surrogate":
                ns = float(r["ns_per_eval"])
                r["speedup_vs_best_exact"] = best_exact_ns / ns
                r["cost_ratio_vs_best_exact"] = ns / best_exact_ns
                r["surrogate_pareto"] = id(r) in surrogate_keep

        # With identical zero error, exactly one exact method should survive.
        exact_pareto = [r for r in exact if r["pareto"]]
        if len(exact_pareto) != 1:
            raise ValueError(
                f"{key}: expected one exact point on Pareto frontier, "
                f"found {len(exact_pareto)}"
            )
        if exact_pareto[0] is not best_exact:
            raise ValueError(
                f"{key}: Pareto exact point is not fastest exact method"
            )


def write(path: Path, rows: list[dict]) -> None:
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
    p.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    p.add_argument("--surrogates", type=Path, default=DEFAULT_SURROGATE)
    p.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED)
    p.add_argument("--pareto-output", type=Path, default=DEFAULT_PARETO)
    a = p.parse_args()

    rows = (
        exact_candidates(read(a.exact))
        + surrogate_candidates(read(a.surrogates))
    )

    annotate(rows)

    rows.sort(
        key=lambda r: (
            r["components"],
            r["batch"],
            float(r["rmse"]),
            float(r["ns_per_eval"]),
            r["candidate_type"],
            str(r["budget"]),
        )
    )

    pareto = [r for r in rows if r["pareto"]]

    write(a.combined_output, rows)
    write(a.pareto_output, pareto)

    print(f"combined candidates={len(rows)}")
    print(f"operating conditions={len(COMPONENTS) * len(BATCHES)}")
    print(f"pareto points={len(pareto)}")


if __name__ == "__main__":
    main()
