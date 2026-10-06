#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"

DIRECT = STUDY / "processed-results/direct-methane.csv"
ACCURACY = STUDY / "processed-results/surrogate-accuracy.csv"
CAL = STUDY / "raw-results/surrogate-performance"

OUT = STUDY / "processed-results/master-performance-terrain.csv"
GLOBAL_PARETO = STUDY / "processed-results/master-pareto-global.csv"
SURROGATE_PARETO = STUDY / "processed-results/surrogate-pareto.csv"


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def selected(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    costs = (
        data.get("policy_costs")
        or data.get("profile", {}).get("policy_costs")
    )
    if not costs:
        return None

    name = (
        data.get("selected_policy")
        or data.get("profile", {}).get("selected_policy")
    )
    if name not in costs:
        name = min(costs, key=lambda k: float(costs[k]))

    return name, float(costs[name])


def nondominated(group):
    keep = set()

    for r in group:
        x = float(r["ns_per_eval"])
        y = float(r["linf"])

        dominated = any(
            (
                float(q["ns_per_eval"]) <= x
                and float(q["linf"]) <= y
            )
            and (
                float(q["ns_per_eval"]) < x
                or float(q["linf"]) < y
            )
            for q in group
            if q is not r
        )

        if not dominated:
            keep.add(id(r))

    return keep


def annotate(rows):
    by_batch = {}

    for r in rows:
        by_batch.setdefault(int(r["batch_size"]), []).append(r)

    for group in by_batch.values():
        exact = [r for r in group if r["exact"]]
        surrogates = [r for r in group if not r["exact"]]

        best_exact = (
            min(float(r["ns_per_eval"]) for r in exact)
            if exact
            else None
        )

        global_keep = nondominated(group)
        surrogate_keep = nondominated(surrogates)

        for r in group:
            r["global_pareto"] = id(r) in global_keep
            r["surrogate_pareto"] = id(r) in surrogate_keep

            r["best_exact_ns_per_eval"] = (
                best_exact if best_exact is not None else ""
            )

            if not r["exact"] and best_exact is not None:
                r["cost_ratio_vs_best_exact"] = (
                    float(r["ns_per_eval"]) / best_exact
                )
            else:
                r["cost_ratio_vs_best_exact"] = ""


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--direct", type=Path, default=DIRECT)
    p.add_argument("--accuracy", type=Path, default=ACCURACY)
    p.add_argument("--calibration-root", type=Path, default=CAL)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument(
        "--global-pareto-output",
        type=Path,
        default=GLOBAL_PARETO,
    )
    p.add_argument(
        "--surrogate-pareto-output",
        type=Path,
        default=SURROGATE_PARETO,
    )

    a = p.parse_args()
    rows = []

    # Exact methods.
    for r in read(a.direct):
        batch = int(r["batch"])

        specs = [
            (
                "CPU scalar",
                "cpu",
                1,
                float(r["scalar_ns"]),
            ),
            (
                "CPU OpenMP",
                "cpu_parallel",
                int(r["best_openmp_threads"]),
                float(r["best_openmp_ns"]),
            ),
            (
                "GPU direct resident",
                "gpu_direct_resident",
                0,
                float(r["cuda_resident_ns"]),
            ),
            (
                "GPU direct E2E",
                "gpu_direct_e2e",
                0,
                float(r["cuda_e2e_ns"]),
            ),
        ]

        for name, backend, threads, ns in specs:
            rows.append(
                dict(
                    method=name,
                    backend=backend,
                    batch_size=batch,
                    budget="",
                    threads=threads,
                    policy="",
                    ns_per_eval=ns,
                    evaluations_per_second=1e9 / ns,
                    rmse=0,
                    p99_abs=0,
                    linf=0,
                    exact=True,
                    feasible=True,
                    evidence="direct benchmark",
                )
            )

    # Surrogate methods.
    accuracy = {
        int(r["budget"]): r
        for r in read(a.accuracy)
    }

    for budget, q in sorted(accuracy.items()):
        calibration_dir = (
            a.calibration_root
            / f"budget-{budget}"
            / "calibration"
        )

        for path in sorted(
            calibration_dir.glob(f"{a.dtype}-b*.json")
        ):
            if path.name.endswith(".trials.json"):
                continue
            if path.name.endswith(".failed.json"):
                continue

            try:
                batch = int(path.stem.split("-b", 1)[1])
            except (IndexError, ValueError):
                continue

            result = selected(path)
            if not result:
                continue

            policy, ns = result

            rows.append(
                dict(
                    method=f"GPU surrogate b{budget}",
                    backend="gpu_surrogate",
                    batch_size=batch,
                    budget=budget,
                    threads=0,
                    policy=policy,
                    ns_per_eval=ns,
                    evaluations_per_second=1e9 / ns,
                    rmse=float(q["rmse"]),
                    p99_abs=float(q["p99_abs"]),
                    linf=float(q["linf"]),
                    exact=False,
                    feasible=True,
                    evidence="calibration selected-policy median",
                )
            )

    annotate(rows)

    fields = [
        "method",
        "backend",
        "batch_size",
        "budget",
        "threads",
        "policy",
        "ns_per_eval",
        "evaluations_per_second",
        "rmse",
        "p99_abs",
        "linf",
        "exact",
        "feasible",
        "global_pareto",
        "surrogate_pareto",
        "best_exact_ns_per_eval",
        "cost_ratio_vs_best_exact",
        "evidence",
    ]

    rows.sort(
        key=lambda r: (
            int(r["batch_size"]),
            str(r["backend"]),
            str(r["budget"]),
        )
    )

    global_rows = [
        r for r in rows
        if r["global_pareto"]
    ]

    surrogate_rows = [
        r for r in rows
        if r["surrogate_pareto"]
    ]

    write(a.output, rows, fields)
    write(a.global_pareto_output, global_rows, fields)
    write(a.surrogate_pareto_output, surrogate_rows, fields)

    print(
        f"records={len(rows)} "
        f"global_pareto={len(global_rows)} "
        f"surrogate_pareto={len(surrogate_rows)}"
    )


if __name__ == "__main__":
    main()
