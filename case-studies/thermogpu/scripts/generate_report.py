#!/usr/bin/env python3
"""Generate the ThermoGPU V1 engineering report from retained derived evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}g}"


def yesno(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct", type=Path, default=STUDY / "processed-results/direct-methane.csv")
    p.add_argument("--accuracy", type=Path, default=STUDY / "processed-results/surrogate-accuracy.csv")
    p.add_argument("--terrain", type=Path, default=STUDY / "processed-results/master-performance-terrain.csv")
    p.add_argument("--feasibility", type=Path, default=STUDY / "processed-results/feasibility-analysis.csv")
    p.add_argument("--summary", type=Path, default=STUDY / "processed-results/study-summary.json")
    p.add_argument("--environment", type=Path, default=STUDY / "raw-results/environment.json")
    p.add_argument("--output", type=Path, default=STUDY / "reports/thermogpu-v1.md")
    args = p.parse_args()

    required = [args.direct, args.accuracy, args.terrain, args.feasibility, args.summary, args.environment]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("cannot generate complete report; missing:\n" + "\n".join(missing))

    direct = read_csv(args.direct)
    accuracy = read_csv(args.accuracy)
    terrain = read_csv(args.terrain)
    feasibility = read_csv(args.feasibility)
    summary = json.loads(args.summary.read_text())
    env = json.loads(args.environment.read_text())

    analysis = summary["analysis"]
    conclusions = summary["conclusions"]
    rep = conclusions["representation_scaling"]
    acc_scale = conclusions["accuracy_cost_scaling"]
    suff = conclusions["measurement_sufficiency"]
    repos = env.get("repositories", {})

    surrogate_rows = [r for r in terrain if r["backend"] == "gpu_surrogate"]
    measured = [r for r in feasibility if r["status"] == "complete"]
    failures = [r for r in feasibility if r["status"] == "resource_failure"]
    deferred = [r for r in feasibility if r["status"] == "deferred"]

    lines = [
        "# ThermoGPU Methane-Z Engineering Surrogate Case Study", "",
        "> Generated from retained case-study evidence. Measured values and conclusions are derived; do not edit them by hand.", "",
        "## Executive result", "",
        "This V1 study tested whether an adaptively constructed TDAR simplicial CPWA surrogate, compiled to an exact CPWA-ReLU DAG and executed on the GPU, provides a computational advantage over direct Peng–Robinson methane compressibility-factor evaluation on the tested hardware and workload range.", "",
        f"It did not. All **{conclusions['dominance_comparisons']}** measured surrogate operating points were dominated by the fastest exact method at the same batch size, and **{conclusions['surrogate_global_pareto_points']}** surrogate points appeared on the conventional global speed/error Pareto frontier. The result is therefore a negative performance result for this measured case, not a claim that CPWA surrogates are generally ineffective.", "",
        "## Scope and frozen design", "",
        "The study compares direct methane-Z evaluation with the pipeline **TDAR → simplicial CPWA → exact CPWA-ReLU DAG → GPU execution**. Surrogate budgets are 16, 32, 64, 128, and 256 points; benchmark batches are 1, 10, 100, 1,000, 10,000, 100,000, and 1,000,000 states. Accuracy is evaluated against the frozen common validation set.", "",
        "The interpretation is empirical and restricted to the tested ThermoGPU methane-Z problem, domain, surrogate construction, execution policies, hardware, budgets, batches, and accuracy range. Fitted power-law exponents below describe measured-range trends; they are not asymptotic complexity theorems.", "",
        "## Frozen provenance", "",
        "| Repository | Version | Commit | Dirty |", "|---|---|---|---|",
    ]
    for name in ("ThermoGPU", "TDAR", "CPWA-ReLU", "case-study"):
        r = repos.get(name, {})
        lines.append(f"| {name} | {r.get('describe', '—')} | `{str(r.get('commit', '—'))[:12]}` | {r.get('dirty', '—')} |")

    lines += ["", "## Exact implementation crossover", "",
              "The preferred exact implementation changes with workload size:", "",
              "| Batch | Preferred exact method | ns/eval | Runner-up | Speedup vs runner-up |", "|---:|---|---:|---|---:|"]
    for r in analysis["exact_crossover"]:
        lines.append(f"| {r['batch_size']:,} | {r['preferred_exact_method']} | {fmt(r['ns_per_eval'])} | {r['runner_up']} | {fmt(r['speedup_vs_runner_up'])}× |")
    lines += ["", "Scalar CPU is best at batches 1 and 10, OpenMP at 100, and resident direct GPU evaluation from batch 1,000 onward. At batch 1,000,000 the resident direct GPU reaches approximately " + fmt(analysis["exact_crossover"][-1]["evaluations_per_second"] / 1e6) + " million evaluations/s.", ""]

    lines += ["## Surrogate accuracy and representation growth", "",
              "| Budget | Simplices | Total DAG nodes | RMSE | p99 abs. error | L∞ error |", "|---:|---:|---:|---:|---:|---:|"]
    dag_by_budget = {int(r["budget"]): r for r in analysis["dag_scaling"]}
    for r in accuracy:
        dag = dag_by_budget[int(r["budget"])]
        lines.append(f"| {r['budget']} | {r['simplices']} | {dag['total_nodes']} | {fmt(r['rmse'])} | {fmt(r['p99_abs'])} | {fmt(r['linf'])} |")
    lines += ["", f"Across the five measured budgets, simplices scale approximately as budget^{fmt(rep['simplices_vs_budget_exponent'])}, while total DAG nodes scale approximately as budget^{fmt(rep['total_nodes_vs_budget_exponent'])}. L∞ error scales approximately as budget^{fmt(acc_scale['linf_vs_budget_exponent'])}. Thus accuracy improves materially, but the exact DAG representation grows substantially faster than the simplex count.", "",
              f"Measured steady-state execution time scales approximately as total_nodes^{fmt(rep['steady_state_time_vs_total_nodes_exponent'])} and as L∞^{fmt(acc_scale['steady_state_time_vs_linf_exponent'])} over this range. These are empirical fits only.", ""]

    lines += ["## Performance and Pareto result", "",
              f"The unified terrain contains **{len(surrogate_rows)}** measured surrogate operating points. **{conclusions['measured_surrogate_points_dominated_by_best_exact_same_batch']}/{conclusions['dominance_comparisons']}** are dominated by the best exact method at the same batch size. The conventional global Pareto set contains **{conclusions['surrogate_global_pareto_points']}** surrogate points.", "",
              "This means the surrogate does not buy speed in exchange for approximation error in the measured regime: an exact implementation is already faster at every directly compared workload size.", "",
              "![Master performance terrain](../figures/master-performance-terrain.png)", "",
              "![Batch-conditioned and conventional Pareto frontiers](../figures/pareto-frontier.png)", ""]

    lines += ["## Feasibility boundary and measurement sufficiency", "",
              f"The frozen 5 × 7 design contains {analysis['measurement_total']} surrogate cells: **{len(measured)} complete**, **{len(failures)} documented resource failure**, and **{len(deferred)} explicitly deferred**.", ""]
    if failures:
        for r in failures:
            lines.append(f"Budget {r['budget']} at batch {int(r['batch_size']):,} failed because of `{r['failure_kind']}` on `{r['hardware']}`; the retained failure record is `{r['evidence']}`.")
        lines.append("")
    if deferred:
        cells = ", ".join(f"B{r['budget']}×{int(r['batch_size']):,}" for r in deferred)
        lines += [f"The deliberately deferred cells are {cells}. They were not required to establish the V1 conclusion because every measured surrogate point was already dominated, no surrogate was globally Pareto-optimal, and no unclassified pending cells remained.", ""]
    lines += [f"Measurement sufficiency for the current V1 conclusion: **{yesno(suff['sufficient_for_current_v1_conclusion'])}**. Additional measurements required for that conclusion: **{yesno(suff['additional_measurements_required_for_current_conclusion'])}**.", ""]

    lines += ["## Engineering interpretation", "",
              "The principal engineering result is that the direct evaluator is already sufficiently efficient that replacing it with this exact CPWA-ReLU realization does not reduce evaluation cost. Increasing surrogate accuracy increases simplicial and DAG complexity, and the execution cost rises enough to erase the usual motivation for a surrogate.", "",
              "The negative result is useful because it identifies where future work should focus: not on assuming that approximation is faster, but on measuring the complete representation-and-execution chain and comparing it against the best workload-conditioned exact implementation. A different thermodynamic problem, dimensionality, hardware target, surrogate representation, or execution strategy may produce a different crossover.", "",
              "## Limitations", "",
              "V1 studies one methane compressibility-factor map on one measured hardware configuration. It does not establish asymptotic scaling, generalize the measured power-law fits beyond the sampled budgets, or prove that other EOS mixtures, higher-dimensional thermodynamic maps, alternative CPWA constructions, reduced DAG representations, accelerators, or execution policies cannot benefit from surrogates. The B256 large-batch cells were deliberately deferred after the V1 conclusion became measurement-sufficient.", "",
              "## Supporting figures", "",
              "![Direct throughput versus batch](../figures/direct-throughput-vs-batch.png)", "",
              "![Surrogate accuracy versus budget](../figures/surrogate-accuracy-vs-budget.png)", "",
              "## Reproduction", "",
              "The scientific results, publication figures, analyses, and this report can be regenerated from retained experimental evidence without rerunning expensive evidence-producing GPU experiments:", "",
              "```bash", "/nvme/Sync/cpwa-relu/.venv/bin/python \\", "  case-studies/thermogpu/scripts/run_study.py \\", "  --config case-studies/thermogpu/configs/study-v1.toml \\", "  --reproduce", "```", "",
              "In reproduction mode, evidence-producing stages are retained and validated; derived processing, figures, Pareto construction, analysis, and report generation are rerun. Legacy surrogate-performance aggregate stages are intentionally omitted because the authoritative V1 performance analysis consumes the retained per-cell calibration evidence directly.", ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
