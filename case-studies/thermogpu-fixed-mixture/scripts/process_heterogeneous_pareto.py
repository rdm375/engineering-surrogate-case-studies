#!/usr/bin/env python3
"""Construct heterogeneous batch-conditioned Pareto frontiers.

This is a post-v1 extension.  It combines:
  * all measured exact implementations,
  * fused-CUDA CPWA,
  * native AVX2 CPWA,
  * native AVX2 + OpenMP 4 CPWA.

Native CPU CPWA evidence exists only for batches 100..1,000,000, so the
heterogeneous frontier is intentionally restricted to those common batches.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
PROCESSED = STUDY / "processed-results"

EXACT = PROCESSED / "exact-performance.csv"
CUDA_CPWA = PROCESSED / "surrogate-performance.csv"
CPU_CPWA = STUDY / "evidence/native-cpu-surrogate/summary.csv"

OUT_CANDIDATES = PROCESSED / "heterogeneous-pareto-candidates.csv"
OUT_FRONTIERS = PROCESSED / "heterogeneous-pareto-frontiers.csv"

COMPONENTS = range(1, 6)
BATCHES = (100, 1000, 10000, 100000, 1000000)
BUDGETS = (16, 24, 32, 48, 64, 96, 128)
CPU_POLICIES = ("avx2", "avx2-omp4")

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
    "dag_nodes",
    "live_slots",
    "registers",
    "spill_store_bytes",
    "spill_load_bytes",
    "stack_frame_bytes",
    "pareto",
    "pareto_exact_cpu",
    "pareto_exact_cpu_omp",
    "pareto_cpwa_cpu",
    "pareto_cpwa_cpu_omp",
    "pareto_exact_gpu",
    "pareto_cpwa_gpu",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def nondominated(rows: list[dict]) -> set[int]:
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

    print(f"wrote {len(rows)} rows: {path}")


exact_raw = read(EXACT)
cuda_raw = read(CUDA_CPWA)
cpu_raw = read(CPU_CPWA)

# CUDA CPWA rows provide the accuracy of each mathematical surrogate.
cuda_lookup = {
    (
        int(r["components"]),
        int(r["budget"]),
        int(r["batch"]),
    ): r
    for r in cuda_raw
    if int(r["batch"]) in BATCHES
}

cpu_lookup = {
    (
        int(r["components"]),
        int(r["budget"]),
        int(r["batch"]),
        r["policy"],
    ): r
    for r in cpu_raw
}

rows = []

# All exact implementations at the five common batches.
for r in exact_raw:
    c = int(r["components"])
    batch = int(r["batch"])

    if batch not in BATCHES:
        continue

    rows.append(
        {
            "components": c,
            "batch": batch,
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
            "dag_nodes": "",
            "live_slots": "",
            "registers": "",
            "spill_store_bytes": "",
            "spill_load_bytes": "",
            "stack_frame_bytes": "",
            "pareto": False,
            "pareto_exact_cpu": False,
            "pareto_exact_cpu_omp": False,
            "pareto_cpwa_cpu": False,
            "pareto_cpwa_cpu_omp": False,
            "pareto_exact_gpu": False,
            "pareto_cpwa_gpu": False,
        }
    )

# All three execution backends for each CPWA artifact.
for c in COMPONENTS:
    for budget in BUDGETS:
        for batch in BATCHES:
            key = (c, budget, batch)

            if key not in cuda_lookup:
                raise RuntimeError(f"missing CUDA CPWA row: {key}")

            g = cuda_lookup[key]
            nodes = int(g["dag_nodes"])
            slots = int(g["live_slots"])

            rows.append(
                {
                    "components": c,
                    "batch": batch,
                    "candidate_type": "surrogate",
                    "method": f"CPWA CUDA B{budget}",
                    "backend": "cpwa-cuda",
                    "budget": budget,
                    "threads": 0,
                    "rmse": float(g["rmse"]),
                    "p99_abs": float(g["p99_abs"]),
                    "max_abs": float(g["max_abs"]),
                    "ns_per_eval": float(g["ns_per_eval"]),
                    "evaluations_per_second":
                        float(g["evaluations_per_second"]),
                    "dag_nodes": nodes,
                    "live_slots": slots,
                    "registers": int(g["registers"]),
                    "spill_store_bytes":
                        int(g["spill_store_bytes"]),
                    "spill_load_bytes":
                        int(g["spill_load_bytes"]),
                    "stack_frame_bytes":
                        int(g["stack_frame_bytes"]),
                    "pareto": False,
            "pareto_exact_cpu": False,
            "pareto_exact_cpu_omp": False,
            "pareto_cpwa_cpu": False,
            "pareto_cpwa_cpu_omp": False,
            "pareto_exact_gpu": False,
            "pareto_cpwa_gpu": False,
                }
            )

            for policy in CPU_POLICIES:
                cpu_key = (c, budget, batch, policy)

                if cpu_key not in cpu_lookup:
                    raise RuntimeError(
                        f"missing native CPU CPWA row: {cpu_key}"
                    )

                p = cpu_lookup[cpu_key]

                if int(p["nodes"]) != nodes:
                    raise RuntimeError(
                        f"DAG-node mismatch for {cpu_key}: "
                        f"{p['nodes']} != {nodes}"
                    )

                if int(p["live_slots"]) != slots:
                    raise RuntimeError(
                        f"live-slot mismatch for {cpu_key}: "
                        f"{p['live_slots']} != {slots}"
                    )

                threads = int(p["threads"])
                name = (
                    f"CPWA AVX2 B{budget}"
                    if policy == "avx2"
                    else f"CPWA AVX2+OMP4 B{budget}"
                )

                rows.append(
                    {
                        "components": c,
                        "batch": batch,
                        "candidate_type": "surrogate",
                        "method": name,
                        "backend": policy,
                        "budget": budget,
                        "threads": threads,
                        "rmse": float(g["rmse"]),
                        "p99_abs": float(g["p99_abs"]),
                        "max_abs": float(g["max_abs"]),
                        "ns_per_eval": float(p["ns_per_eval"]),
                        "evaluations_per_second":
                            1.0e9 / float(p["ns_per_eval"]),
                        "dag_nodes": nodes,
                        "live_slots": slots,
                        "registers": "",
                        "spill_store_bytes": "",
                        "spill_load_bytes": "",
                        "stack_frame_bytes": "",
                        "pareto": False,
            "pareto_exact_cpu": False,
            "pareto_exact_cpu_omp": False,
            "pareto_cpwa_cpu": False,
            "pareto_cpwa_cpu_omp": False,
            "pareto_exact_gpu": False,
            "pareto_cpwa_gpu": False,
                    }
                )


groups = defaultdict(list)
groups = defaultdict(list)

for r in rows:
    groups[(r["components"], r["batch"])].append(r)

expected_groups = {
    (c, batch)
    for c in COMPONENTS
    for batch in BATCHES
}

if set(groups) != expected_groups:
    raise RuntimeError("incomplete heterogeneous Pareto coverage")


def family_rows(
    group: list[dict],
    family: str,
) -> list[dict]:
    """Return candidates belonging to one execution family."""

    if family == "exact_cpu":
        return [
            r for r in group
            if (
                r["candidate_type"] == "exact"
                and r["backend"] == "scalar"
            )
        ]

    if family == "exact_cpu_omp":
        return [
            r for r in group
            if (
                r["candidate_type"] == "exact"
                and r["backend"] == "openmp"
            )
        ]

    if family == "cpwa_cpu":
        return [
            r for r in group
            if (
                r["candidate_type"] == "surrogate"
                and r["backend"] == "avx2"
            )
        ]

    if family == "cpwa_cpu_omp":
        return [
            r for r in group
            if (
                r["candidate_type"] == "surrogate"
                and r["backend"] == "avx2-omp4"
            )
        ]

    if family == "exact_gpu":
        return [
            r for r in group
            if (
                r["candidate_type"] == "exact"
                and r["backend"] in {
                    "cuda_fixed_resident",
                    "cuda_generic_resident",
                }
            )
        ]

    if family == "cpwa_gpu":
        return [
            r for r in group
            if (
                r["candidate_type"] == "surrogate"
                and r["backend"] == "cpwa-cuda"
            )
        ]

    raise ValueError(f"unknown execution family: {family}")


FAMILY_FIELDS = {
    "exact_cpu": "pareto_exact_cpu",
    "exact_cpu_omp": "pareto_exact_cpu_omp",
    "cpwa_cpu": "pareto_cpwa_cpu",
    "cpwa_cpu_omp": "pareto_cpwa_cpu_omp",
    "exact_gpu": "pareto_exact_gpu",
    "cpwa_gpu": "pareto_cpwa_gpu",
}


for key, group in groups.items():
    exact = [
        r for r in group
        if r["candidate_type"] == "exact"
    ]
    surrogate = [
        r for r in group
        if r["candidate_type"] == "surrogate"
    ]

    if len(exact) != 7:
        raise RuntimeError(
            f"{key}: expected 7 exact candidates, "
            f"found {len(exact)}"
        )

    if len(surrogate) != 21:
        raise RuntimeError(
            f"{key}: expected 21 surrogate candidates, "
            f"found {len(surrogate)}"
        )

    if len(group) != 28:
        raise RuntimeError(
            f"{key}: expected 28 total candidates, "
            f"found {len(group)}"
        )

    # All three CPWA backends at a given budget must represent the
    # same mathematical approximation.
    for budget in BUDGETS:
        same = [
            r for r in surrogate
            if int(r["budget"]) == budget
        ]

        if len(same) != 3:
            raise RuntimeError(
                f"{key}, B{budget}: expected 3 CPWA backends, "
                f"found {len(same)}"
            )

        accuracy = {
            (
                float(r["rmse"]),
                float(r["p99_abs"]),
                float(r["max_abs"]),
            )
            for r in same
        }

        if len(accuracy) != 1:
            raise RuntimeError(
                f"{key}, B{budget}: CPWA accuracy mismatch"
            )

    # Global heterogeneous frontier.
    global_keep = nondominated(group)

    for r in group:
        r["pareto"] = id(r) in global_keep

    # Independent execution-family frontiers.
    for family, field in FAMILY_FIELDS.items():
        members = family_rows(group, family)

        expected = {
            "exact_cpu": 1,
            "exact_cpu_omp": 4,
            "cpwa_cpu": 7,
            "cpwa_cpu_omp": 7,
            "exact_gpu": 2,
            "cpwa_gpu": 7,
        }[family]

        if len(members) != expected:
            raise RuntimeError(
                f"{key}: {family} expected {expected} candidates, "
                f"found {len(members)}"
            )

        keep = nondominated(members)

        for r in members:
            r[field] = id(r) in keep

    # Exact families live at RMSE=0.  Their family Pareto set must
    # therefore contain exactly one fastest implementation.
    for family in (
        "exact_cpu",
        "exact_cpu_omp",
        "exact_gpu",
    ):
        field = FAMILY_FIELDS[family]
        count = sum(
            r[field]
            for r in family_rows(group, family)
        )

        if count != 1:
            raise RuntimeError(
                f"{key}: {family} has {count} Pareto points"
            )

    # The global exact endpoint must likewise be the fastest exact
    # implementation at this operating point.
    exact_global = [
        r for r in exact
        if r["pareto"]
    ]

    if len(exact_global) != 1:
        raise RuntimeError(
            f"{key}: expected one global exact endpoint, "
            f"found {len(exact_global)}"
        )


rows.sort(
    key=lambda r: (
        int(r["components"]),
        int(r["batch"]),
        float(r["rmse"]),
        float(r["ns_per_eval"]),
        r["method"],
    )
)

frontiers = [
    r for r in rows
    if r["pareto"]
]

write(OUT_CANDIDATES, rows)
write(OUT_FRONTIERS, frontiers)
