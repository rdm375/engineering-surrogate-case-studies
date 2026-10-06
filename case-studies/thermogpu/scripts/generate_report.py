#!/usr/bin/env python3
"""Generate the ThermoGPU V1 report from retained processed evidence."""
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


def fmt(x: str | float, digits=4) -> str:
    return f"{float(x):.{digits}g}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--direct", type=Path, default=STUDY / "processed-results/direct-methane.csv")
    p.add_argument("--accuracy", type=Path, default=STUDY / "processed-results/surrogate-accuracy.csv")
    p.add_argument("--performance", type=Path, default=STUDY / "processed-results/surrogate-performance.csv")
    p.add_argument("--environment", type=Path, default=STUDY / "raw-results/environment.json")
    p.add_argument("--output", type=Path, default=STUDY / "reports/thermogpu-v1.md")
    args = p.parse_args()
    required = [args.direct, args.accuracy, args.performance, args.environment]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit("cannot generate complete report; missing:\n" + "\n".join(missing))

    direct = read_csv(args.direct)
    accuracy = read_csv(args.accuracy)
    perf = read_csv(args.performance)
    env = json.loads(args.environment.read_text())
    repos = env.get("repositories", {})

    lines = [
        "# ThermoGPU Methane-Z Engineering Surrogate Case Study", "",
        "> Generated from retained case-study evidence. Do not edit measured values by hand.", "",
        "## Scope", "",
        "This V1 study compares direct Peng–Robinson methane compressibility-factor evaluation with TDAR → simplicial CPWA → exact CPWA-ReLU surrogate evaluation over the frozen temperature/pressure domain.", "",
        "## Frozen provenance", "",
        "| Repository | Version | Commit | Dirty |", "|---|---|---|---|",
    ]
    for name in ("ThermoGPU", "TDAR", "CPWA-ReLU", "case-study"):
        r = repos.get(name, {})
        lines.append(f"| {name} | {r.get('describe','—')} | `{str(r.get('commit','—'))[:12]}` | {r.get('dirty','—')} |")

    lines += ["", "## Direct-model performance", "", "| Batch | Scalar ns/state | Best OpenMP ns/state | Threads | CUDA resident ns/state | CUDA end-to-end ns/state |", "|---:|---:|---:|---:|---:|---:|"]
    for r in direct:
        lines.append(f"| {r['batch']} | {fmt(r['scalar_ns'])} | {fmt(r['best_openmp_ns'])} | {r['best_openmp_threads']} | {fmt(r['cuda_resident_ns'])} | {fmt(r['cuda_e2e_ns'])} |")

    lines += ["", "## Surrogate accuracy", "", "| Budget | Simplices | RMSE | p99 abs. error | L∞ error | TDAR build s |", "|---:|---:|---:|---:|---:|---:|"]
    for r in accuracy:
        lines.append(f"| {r['budget']} | {r['simplices']} | {fmt(r['rmse'])} | {fmt(r['p99_abs'])} | {fmt(r['linf'])} | {fmt(r['tdar_seconds'])} |")

    lines += ["", "## Exact CPWA-ReLU GPU performance", "", "The table below reports the empirically selected execution policy at each measured budget/batch operating point.", "", "| Budget | Batch | Policy | ns/eval | M eval/s |", "|---:|---:|---|---:|---:|"]
    for r in sorted(perf, key=lambda x: (int(x['budget']), int(x['batch']))):
        lines.append(f"| {r['budget']} | {r['batch']} | `{r['policy']}` | {fmt(r['ns_per_eval'])} | {float(r['evaluations_per_second'])/1e6:.4g} |")

    lines += ["", "## Figures", "", "![Direct throughput versus batch](../figures/direct-throughput-vs-batch.png)", "", "![Surrogate accuracy versus budget](../figures/surrogate-accuracy-vs-budget.png)", "", "![Exact CPWA-ReLU throughput versus batch](../figures/surrogate-throughput-vs-batch.png)", "", "## Reproduction", "", "Run the resumable local pipeline from the repository root:", "", "```bash", "python case-studies/thermogpu/scripts/run_study.py", "```", "", "GPU calibration and surrogate benchmarking deliberately use the CPWA-ReLU virtual-environment Python so the frozen GPU-enabled JAX installation is used. Completed stages are skipped unless `--force` is supplied.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
