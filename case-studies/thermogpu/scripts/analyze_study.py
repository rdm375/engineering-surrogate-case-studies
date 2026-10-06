#!/usr/bin/env python3
"""Derive reusable engineering conclusions from standardized case-study evidence."""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"
PROCESSED = STUDY / "processed-results"
RAW = STUDY / "raw-results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(path)


def exact_crossover(direct: list[dict[str, str]]) -> list[dict]:
    methods = [
        ("CPU scalar", "scalar_ns"),
        ("CPU OpenMP", "best_openmp_ns"),
        ("GPU direct resident", "cuda_resident_ns"),
        ("GPU direct E2E", "cuda_e2e_ns"),
    ]
    out = []
    for r in direct:
        costs = {name: float(r[col]) for name, col in methods}
        ranked = sorted(costs.items(), key=lambda x: x[1])
        best, best_ns = ranked[0]
        second, second_ns = ranked[1]
        out.append({
            "batch_size": int(r["batch"]),
            "preferred_exact_method": best,
            "ns_per_eval": best_ns,
            "evaluations_per_second": 1e9 / best_ns,
            "runner_up": second,
            "runner_up_ns_per_eval": second_ns,
            "speedup_vs_runner_up": second_ns / best_ns,
            "speedup_vs_scalar": float(r["scalar_ns"]) / best_ns,
        })
    return out


def accuracy_cost(master: list[dict[str, str]]) -> list[dict]:
    out = []
    for r in master:
        if r["exact"].lower() == "true":
            continue
        out.append({
            "budget": int(r["budget"]),
            "batch_size": int(r["batch_size"]),
            "policy": r["policy"],
            "rmse": float(r["rmse"]),
            "p99_abs": float(r["p99_abs"]),
            "linf": float(r["linf"]),
            "ns_per_eval": float(r["ns_per_eval"]),
            "evaluations_per_second": float(r["evaluations_per_second"]),
            "best_exact_ns_per_eval": float(r["best_exact_ns_per_eval"]),
            "cost_ratio_vs_best_exact": float(r["cost_ratio_vs_best_exact"]),
            "surrogate_pareto": r["surrogate_pareto"],
            "global_pareto": r["global_pareto"],
        })
    return sorted(out, key=lambda r: (r["budget"], r["batch_size"]))


def dominance(master: list[dict[str, str]]) -> list[dict]:
    out = []
    by_batch: dict[int, list[dict[str, str]]] = {}
    for r in master:
        by_batch.setdefault(int(r["batch_size"]), []).append(r)
    for batch, rows in sorted(by_batch.items()):
        exact = [r for r in rows if r["exact"].lower() == "true"]
        sur = [r for r in rows if r["exact"].lower() != "true"]
        if not exact:
            continue
        best = min(exact, key=lambda r: float(r["ns_per_eval"]))
        best_ns = float(best["ns_per_eval"])
        for r in sur:
            ns = float(r["ns_per_eval"])
            err = float(r["linf"])
            exact_dominates = best_ns <= ns and 0.0 <= err and (best_ns < ns or err > 0)
            surrogate_dominates = ns <= best_ns and err <= 0.0 and (ns < best_ns or err < 0.0)
            out.append({
                "batch_size": batch,
                "budget": int(r["budget"]),
                "surrogate_ns_per_eval": ns,
                "linf": err,
                "best_exact_method": best["method"],
                "best_exact_ns_per_eval": best_ns,
                "cost_ratio_vs_best_exact": ns / best_ns,
                "best_exact_dominates_surrogate": exact_dominates,
                "surrogate_dominates_best_exact": surrogate_dominates,
                "surrogate_global_pareto": r["global_pareto"],
            })
    return out


def artifact_stats(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        metadata = json.loads(z.read("metadata.json"))
        arrays = np.load(io.BytesIO(z.read("dag.npz")))
        kinds = Counter(int(x) for x in arrays["kinds"].tolist())
    # Native format currently uses 0=affine, 1=min, 2=max.
    return {
        "affine_nodes": kinds.get(0, 0),
        "min_nodes": kinds.get(1, 0),
        "max_nodes": kinds.get(2, 0),
        "total_nodes": int(metadata["nodes"]),
        "affine_pieces": int(metadata["affine_pieces"]),
        "lower_roots": int(metadata["lower_roots"]),
        "artifact_bytes": path.stat().st_size,
    }


def dag_scaling(accuracy: list[dict[str, str]], perf_root: Path) -> list[dict]:
    out = []
    for q in accuracy:
        budget = int(q["budget"])
        base = perf_root / f"budget-{budget}"
        artifacts = sorted((base / "artifacts").glob("*.cpwa"))
        if not artifacts:
            continue
        stats = artifact_stats(artifacts[0])
        build_path = base / "build-native.json"
        build = json.loads(build_path.read_text()) if build_path.exists() else {}
        trials = []
        for p in sorted((base / "calibration").glob("*.trials.json")):
            try:
                d = json.loads(p.read_text())
                selected = d.get("selected_policy")
                phase = d.get("phase_timings", {}).get(selected, {})
                if phase:
                    trials.append((int(d["profile"]["batch_size"]), selected, phase))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        # Use batch=1000 when available as a common operating point; otherwise largest measured batch.
        chosen = next((x for x in trials if x[0] == 1000), max(trials, default=None, key=lambda x: x[0]))
        row = {
            "budget": budget,
            "points": int(q["points"]),
            "simplices": int(q["simplices"]),
            "rmse": float(q["rmse"]),
            "p99_abs": float(q["p99_abs"]),
            "linf": float(q["linf"]),
            "tdar_seconds": float(q["tdar_seconds"]),
            **stats,
            "native_build_wall_seconds": build.get("wall_seconds", ""),
            "reference_batch_size": "",
            "reference_policy": "",
            "lowering_seconds": "",
            "compile_first_execution_seconds": "",
            "steady_state_ns_per_eval": "",
        }
        if chosen:
            batch, policy, phase = chosen
            row.update({
                "reference_batch_size": batch,
                "reference_policy": policy,
                "lowering_seconds": phase.get("lowering_seconds", ""),
                "compile_first_execution_seconds": phase.get("compile_first_execution_seconds", ""),
                "steady_state_ns_per_eval": phase.get("ns_per_eval", ""),
            })
        out.append(row)
    return out


def feasibility(completeness: list[dict[str, str]]) -> list[dict]:
    return [{
        "budget": int(r["budget"]),
        "batch_size": int(r["batch_size"]),
        "status": r["status"],
        "evidence": r.get("evidence", ""),
    } for r in completeness]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-name", default="thermogpu-methane-z-v1")
    p.add_argument("--direct", type=Path, default=PROCESSED / "direct-methane.csv")
    p.add_argument("--accuracy", type=Path, default=PROCESSED / "surrogate-accuracy.csv")
    p.add_argument("--master", type=Path, default=PROCESSED / "master-performance-terrain.csv")
    p.add_argument("--completeness", type=Path, default=PROCESSED / "experiment-completeness.csv")
    p.add_argument("--performance-root", type=Path, default=RAW / "surrogate-performance")
    p.add_argument("--output-dir", type=Path, default=PROCESSED)
    a = p.parse_args()

    direct = read_csv(a.direct)
    accuracy = read_csv(a.accuracy)
    master = read_csv(a.master)
    complete = read_csv(a.completeness)

    cross = exact_crossover(direct)
    costs = accuracy_cost(master)
    dom = dominance(master)
    dags = dag_scaling(accuracy, a.performance_root)
    feas = feasibility(complete)

    write_csv(a.output_dir / "exact-crossover.csv", cross, list(cross[0]))
    write_csv(a.output_dir / "surrogate-accuracy-cost.csv", costs, list(costs[0]))
    write_csv(a.output_dir / "dominance-analysis.csv", dom, list(dom[0]))
    write_csv(a.output_dir / "dag-scaling.csv", dags, list(dags[0]))
    write_csv(a.output_dir / "feasibility-analysis.csv", feas, list(feas[0]))

    status_counts = Counter(r["status"] for r in feas)
    global_surrogate = sum(str(r["global_pareto"]).lower() == "true" for r in costs)
    dominated = sum(bool(r["best_exact_dominates_surrogate"]) for r in dom)
    summary = {
        "schema_version": 1,
        "study": a.study_name,
        "analysis": {
            "exact_crossover": cross,
            "surrogate_measurements": len(costs),
            "surrogate_global_pareto_points": global_surrogate,
            "surrogate_points_dominated_by_best_exact_same_batch": dominated,
            "dominance_comparisons": len(dom),
            "dag_scaling": dags,
            "measurement_status": dict(sorted(status_counts.items())),
            "measurement_total": len(feas),
        },
    }
    summary_path = a.output_dir / "study-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)
    print(f"exact_batches={len(cross)} surrogate_measurements={len(costs)} dominated_same_batch={dominated}/{len(dom)} global_surrogate_pareto={global_surrogate}")


if __name__ == "__main__":
    main()
