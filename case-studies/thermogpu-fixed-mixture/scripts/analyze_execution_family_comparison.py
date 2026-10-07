#!/usr/bin/env python3
"""Compare CPWA execution families at identical approximation accuracy.

For every (components, batch, budget), the AVX2, AVX2+OpenMP4, and fused
CUDA implementations execute the same mathematical CPWA approximation.
This permits a direct execution-backend comparison without interpolation
or accuracy matching.

This is a post-v1 derived analysis.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
PROCESSED = STUDY / "processed-results"

INPUT = PROCESSED / "heterogeneous-pareto-candidates.csv"
OUTPUT = PROCESSED / "execution-family-comparison.csv"

COMPONENTS = range(1, 6)
BATCHES = (100, 1000, 10000, 100000, 1000000)
BUDGETS = (16, 24, 32, 48, 64, 96, 128)

FIELDS = [
    "components",
    "batch",
    "budget",
    "rmse",
    "dag_nodes",
    "live_slots",
    "avx2_ns_per_eval",
    "avx2_omp4_ns_per_eval",
    "cuda_ns_per_eval",
    "fastest_cpwa_backend",
    "fastest_cpwa_ns_per_eval",
    "avx2_cost_ratio_vs_best_cpwa",
    "avx2_omp4_cost_ratio_vs_best_cpwa",
    "cuda_cost_ratio_vs_best_cpwa",
    "best_exact_method",
    "best_exact_backend",
    "best_exact_ns_per_eval",
    "best_cpwa_speedup_vs_best_exact",
    "global_winner_type",
    "global_winner_method",
    "global_winner_backend",
    "global_winner_ns_per_eval",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows: {path}")


rows = read(INPUT)

groups = defaultdict(list)
for r in rows:
    groups[(int(r["components"]), int(r["batch"]))].append(r)

expected_groups = {
    (c, batch)
    for c in COMPONENTS
    for batch in BATCHES
}

if set(groups) != expected_groups:
    raise RuntimeError("incomplete (components, batch) coverage")

output = []

for c in COMPONENTS:
    for batch in BATCHES:
        group = groups[(c, batch)]

        exact = [
            r for r in group
            if r["candidate_type"] == "exact"
        ]

        if len(exact) != 7:
            raise RuntimeError(
                f"C{c}, batch {batch}: expected 7 exact rows, "
                f"found {len(exact)}"
            )

        best_exact = min(
            exact,
            key=lambda r: float(r["ns_per_eval"]),
        )

        for budget in BUDGETS:
            cpwa = [
                r for r in group
                if (
                    r["candidate_type"] == "surrogate"
                    and int(r["budget"]) == budget
                )
            ]

            if len(cpwa) != 3:
                raise RuntimeError(
                    f"C{c}, batch {batch}, B{budget}: "
                    f"expected 3 CPWA rows, found {len(cpwa)}"
                )

            by_backend = {r["backend"]: r for r in cpwa}

            expected_backends = {
                "avx2",
                "avx2-omp4",
                "cpwa-cuda",
            }

            if set(by_backend) != expected_backends:
                raise RuntimeError(
                    f"C{c}, batch {batch}, B{budget}: "
                    f"unexpected CPWA backends {sorted(by_backend)}"
                )

            # These must be the same mathematical approximation.
            accuracy = {
                (
                    float(r["rmse"]),
                    float(r["p99_abs"]),
                    float(r["max_abs"]),
                )
                for r in cpwa
            }

            if len(accuracy) != 1:
                raise RuntimeError(
                    f"C{c}, batch {batch}, B{budget}: "
                    "accuracy differs across CPWA backends"
                )

            nodes = {int(r["dag_nodes"]) for r in cpwa}
            slots = {int(r["live_slots"]) for r in cpwa}

            if len(nodes) != 1 or len(slots) != 1:
                raise RuntimeError(
                    f"C{c}, batch {batch}, B{budget}: "
                    "DAG metadata differs across CPWA backends"
                )

            avx2 = by_backend["avx2"]
            omp4 = by_backend["avx2-omp4"]
            cuda = by_backend["cpwa-cuda"]

            costs = {
                "avx2": float(avx2["ns_per_eval"]),
                "avx2-omp4": float(omp4["ns_per_eval"]),
                "cpwa-cuda": float(cuda["ns_per_eval"]),
            }

            fastest_backend = min(costs, key=costs.get)
            fastest_cost = costs[fastest_backend]

            # "Global winner" here means the minimum execution cost
            # available at this exact CPWA accuracy, including exact
            # physics at RMSE=0.  Exact physics is admissible because
            # zero error satisfies every positive error tolerance.
            if float(best_exact["ns_per_eval"]) <= fastest_cost:
                global_type = "exact"
                global_method = best_exact["method"]
                global_backend = best_exact["backend"]
                global_cost = float(best_exact["ns_per_eval"])
            else:
                winner = by_backend[fastest_backend]
                global_type = "surrogate"
                global_method = winner["method"]
                global_backend = winner["backend"]
                global_cost = fastest_cost

            output.append(
                {
                    "components": c,
                    "batch": batch,
                    "budget": budget,
                    "rmse": float(avx2["rmse"]),
                    "dag_nodes": next(iter(nodes)),
                    "live_slots": next(iter(slots)),
                    "avx2_ns_per_eval": costs["avx2"],
                    "avx2_omp4_ns_per_eval":
                        costs["avx2-omp4"],
                    "cuda_ns_per_eval": costs["cpwa-cuda"],
                    "fastest_cpwa_backend": fastest_backend,
                    "fastest_cpwa_ns_per_eval": fastest_cost,
                    "avx2_cost_ratio_vs_best_cpwa":
                        costs["avx2"] / fastest_cost,
                    "avx2_omp4_cost_ratio_vs_best_cpwa":
                        costs["avx2-omp4"] / fastest_cost,
                    "cuda_cost_ratio_vs_best_cpwa":
                        costs["cpwa-cuda"] / fastest_cost,
                    "best_exact_method": best_exact["method"],
                    "best_exact_backend": best_exact["backend"],
                    "best_exact_ns_per_eval":
                        float(best_exact["ns_per_eval"]),
                    "best_cpwa_speedup_vs_best_exact":
                        float(best_exact["ns_per_eval"])
                        / fastest_cost,
                    "global_winner_type": global_type,
                    "global_winner_method": global_method,
                    "global_winner_backend": global_backend,
                    "global_winner_ns_per_eval": global_cost,
                }
            )

if len(output) != 175:
    raise RuntimeError(
        f"expected 175 comparison rows, found {len(output)}"
    )

write(OUTPUT, output)
