#!/usr/bin/env python3
"""Compare exact and CPWA execution across CPU and CUDA backends.

This is an additive post-v1 analysis.  It consumes the frozen Case Study 2
processed results plus retained native-CPU CPWA evidence.  It does not modify
the v1 evidence or processed datasets.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
PROCESSED = STUDY / "processed-results"
CPU_EVIDENCE = STUDY / "evidence/native-cpu-surrogate/summary.csv"

OUT = PROCESSED / "heterogeneous-execution.csv"
SUMMARY = PROCESSED / "heterogeneous-summary.csv"

TARGETS = (1e-3, 7.5e-4, 5e-4, 4e-4)
BATCHES = (100, 1000, 10000, 100000, 1000000)


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)


def f(x):
    return float(x)


def i(x):
    return int(x)


exact = read_csv(PROCESSED / "exact-performance.csv")
surrogate = read_csv(PROCESSED / "surrogate-performance.csv")
cpu = read_csv(CPU_EVIDENCE)

# Accuracy is independent of execution batch.  Collapse to one row per C/B.
accuracy = {}
for r in surrogate:
    key = (i(r["components"]), i(r["budget"]))
    value = f(r["rmse"])
    old = accuracy.setdefault(key, value)
    if abs(old - value) > 1e-15:
        raise RuntimeError(f"inconsistent RMSE for {key}: {old} vs {value}")

# CUDA surrogate timing.
cuda_surrogate = {
    (i(r["components"]), i(r["budget"]), i(r["batch"])): f(r["ns_per_eval"])
    for r in surrogate
}

# Native CPU surrogate timing.
cpu_surrogate = {
    (
        i(r["components"]),
        i(r["budget"]),
        i(r["batch"]),
        r["policy"],
    ): f(r["ns_per_eval"])
    for r in cpu
}

# Exact candidates by C/batch.
exact_by = {}
for r in exact:
    key = (i(r["components"]), i(r["batch"]))
    exact_by.setdefault(key, []).append(r)


def classify_exact(r):
    backend = r["backend"]
    if backend.startswith("cuda"):
        return "exact-cuda"
    return "exact-cpu"


rows = []

for components in range(1, 6):
    budgets = sorted(
        b for (c, b), rmse in accuracy.items()
        if c == components
    )

    for target in TARGETS:
        eligible = [
            b for b in budgets
            if accuracy[(components, b)] <= target
        ]

        if not eligible:
            continue

        # Preserve v1 methodology: smallest measured budget satisfying target.
        budget = min(eligible)
        rmse = accuracy[(components, budget)]

        for batch in BATCHES:
            exact_candidates = exact_by[(components, batch)]

            best_exact_cpu = min(
                (r for r in exact_candidates
                 if classify_exact(r) == "exact-cpu"),
                key=lambda r: f(r["ns_per_eval"]),
            )
            best_exact_cuda = min(
                (r for r in exact_candidates
                 if classify_exact(r) == "exact-cuda"),
                key=lambda r: f(r["ns_per_eval"]),
            )

            candidates = [
                {
                    "execution": "exact-cpu",
                    "method": best_exact_cpu["method"],
                    "ns": f(best_exact_cpu["ns_per_eval"]),
                },
                {
                    "execution": "exact-cuda",
                    "method": best_exact_cuda["method"],
                    "ns": f(best_exact_cuda["ns_per_eval"]),
                },
                {
                    "execution": "cpwa-cpu-avx2",
                    "method": "CPWA AVX2",
                    "ns": cpu_surrogate[
                        (components, budget, batch, "avx2")
                    ],
                },
                {
                    "execution": "cpwa-cpu-avx2-omp4",
                    "method": "CPWA AVX2 + OpenMP 4",
                    "ns": cpu_surrogate[
                        (components, budget, batch, "avx2-omp4")
                    ],
                },
                {
                    "execution": "cpwa-cuda",
                    "method": "CPWA fused CUDA",
                    "ns": cuda_surrogate[
                        (components, budget, batch)
                    ],
                },
            ]

            winner = min(candidates, key=lambda x: x["ns"])
            exact_best = min(
                f(best_exact_cpu["ns_per_eval"]),
                f(best_exact_cuda["ns_per_eval"]),
            )

            by_name = {x["execution"]: x["ns"] for x in candidates}

            rows.append({
                "components": components,
                "rmse_target": target,
                "selected_budget": budget,
                "actual_rmse": rmse,
                "batch": batch,
                "exact_cpu_method": best_exact_cpu["method"],
                "exact_cpu_ns": by_name["exact-cpu"],
                "exact_cuda_method": best_exact_cuda["method"],
                "exact_cuda_ns": by_name["exact-cuda"],
                "cpwa_avx2_ns": by_name["cpwa-cpu-avx2"],
                "cpwa_avx2_omp4_ns": by_name["cpwa-cpu-avx2-omp4"],
                "cpwa_cuda_ns": by_name["cpwa-cuda"],
                "winner": winner["execution"],
                "winner_method": winner["method"],
                "winner_ns": winner["ns"],
                "speedup_vs_best_exact": exact_best / winner["ns"],
                "cpu_cpwa_speedup_vs_exact_cpu":
                    by_name["exact-cpu"] /
                    min(
                        by_name["cpwa-cpu-avx2"],
                        by_name["cpwa-cpu-avx2-omp4"],
                    ),
                "cpu_cpwa_speedup_vs_exact_cuda":
                    by_name["exact-cuda"] /
                    min(
                        by_name["cpwa-cpu-avx2"],
                        by_name["cpwa-cpu-avx2-omp4"],
                    ),
            })


# One compact row per C/target at the largest measured batch.
summary = []
for r in rows:
    if i(r["batch"]) != 1_000_000:
        continue
    summary.append({
        "components": r["components"],
        "rmse_target": r["rmse_target"],
        "selected_budget": r["selected_budget"],
        "actual_rmse": r["actual_rmse"],
        "winner": r["winner"],
        "winner_ns": r["winner_ns"],
        "speedup_vs_best_exact": r["speedup_vs_best_exact"],
        "exact_cpu_ns": r["exact_cpu_ns"],
        "exact_cuda_ns": r["exact_cuda_ns"],
        "cpwa_avx2_ns": r["cpwa_avx2_ns"],
        "cpwa_avx2_omp4_ns": r["cpwa_avx2_omp4_ns"],
        "cpwa_cuda_ns": r["cpwa_cuda_ns"],
        "cpu_cpwa_speedup_vs_exact_cpu":
            r["cpu_cpwa_speedup_vs_exact_cpu"],
        "cpu_cpwa_speedup_vs_exact_cuda":
            r["cpu_cpwa_speedup_vs_exact_cuda"],
    })


write_csv(OUT, rows)
write_csv(SUMMARY, summary)

print(f"wrote {len(rows)} rows: {OUT}")
print(f"wrote {len(summary)} rows: {SUMMARY}")

REGIMES = PROCESSED / "execution-regimes.csv"

regimes = []
for components in range(1, 6):
    for target in TARGETS:
        q = [
            r for r in rows
            if int(r["components"]) == components
            and float(r["rmse_target"]) == target
        ]
        q.sort(key=lambda r: int(r["batch"]))

        cpu_wins = []
        cuda_wins = []

        for r in q:
            cpu_ns = min(
                float(r["cpwa_avx2_ns"]),
                float(r["cpwa_avx2_omp4_ns"]),
            )
            cuda_ns = float(r["cpwa_cuda_ns"])

            if cpu_ns < cuda_ns:
                cpu_wins.append(int(r["batch"]))
            elif cuda_ns < cpu_ns:
                cuda_wins.append(int(r["batch"]))

        regimes.append({
            "components": components,
            "rmse_target": target,
            "selected_budget": q[0]["selected_budget"],
            "actual_rmse": q[0]["actual_rmse"],
            "largest_measured_cpu_win_batch":
                max(cpu_wins) if cpu_wins else "",
            "smallest_measured_cuda_win_batch":
                min(cuda_wins) if cuda_wins else "",
            "crossover_bracket":
                (
                    f"{max(cpu_wins)}..{min(cuda_wins)}"
                    if cpu_wins and cuda_wins
                    else ""
                ),
        })

write_csv(REGIMES, regimes)
print(f"wrote {len(regimes)} rows: {REGIMES}")
