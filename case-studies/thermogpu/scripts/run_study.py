#!/usr/bin/env python3
"""Run the frozen ThermoGPU V1 case study as a resumable local pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"
SCRIPTS = STUDY / "scripts"
RAW = STUDY / "raw-results"
PROCESSED = STUDY / "processed-results"
FIGURES = STUDY / "figures"
REPORTS = STUDY / "reports"
MANIFEST = RAW / "pipeline-manifest.json"


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: list[str]
    outputs: tuple[Path, ...]


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def canonical_direct_stem(batches: list[int]) -> str:
    return "scaling-b" + "-".join(map(str, batches))


def build_stages(args) -> list[Stage]:
    cfg = load_config(args.config)
    budgets = [int(x) for x in cfg["surrogate"]["budgets"]]
    batches = [int(x) for x in cfg["benchmark"]["batch_sizes"]]
    validation_points = int(cfg["validation"]["points"])
    seed = int(cfg["validation"]["seed"])
    initial = int(cfg["surrogate"]["initial_points"])
    direct_stem = canonical_direct_stem(batches)
    py = str(Path(sys.executable).resolve())
    cpwa_py = str(args.cpwa_python)
    common_cpwa = ["--cpwa-relu-root", str(args.cpwa_relu_root.resolve())]
    budget_args = [str(x) for x in budgets]
    batch_args = [str(x) for x in batches]

    return [
        Stage("environment", "capture repository and hardware provenance",
              [py, str(SCRIPTS / "collect_environment.py")],
              (RAW / "environment.json",)),
        Stage("direct", "collect canonical ThermoGPU direct scaling evidence",
              [py, str(SCRIPTS / "collect_direct.py"), "--thermogpu-root", str(args.thermogpu_root.resolve()), "--batch", *batch_args],
              (RAW / "direct" / f"{direct_stem}.json",)),
        Stage("direct-process", "normalize direct methane results",
              [py, str(SCRIPTS / "process_direct.py")],
              (PROCESSED / "direct-methane.csv",)),
        Stage("direct-plot", "plot direct throughput scaling",
              [py, str(SCRIPTS / "plot_direct.py")],
              (FIGURES / "direct-throughput-vs-batch.png",)),
        Stage("surrogates", "build and validate the frozen TDAR surrogate sweep",
              [py, str(SCRIPTS / "build_surrogates.py"), "--budgets", *budget_args, "--initial", str(initial), "--validation-points", str(validation_points), "--seed", str(seed), "--tdar-root", str(args.tdar_root.resolve()), "--thermogpu-root", str(args.thermogpu_root.resolve())],
              (RAW / "surrogates" / "normalized.json", RAW / "surrogates" / "validation-points.npy", RAW / "surrogates" / "validation-truth.npy")),
        Stage("surrogate-process", "normalize TDAR accuracy results",
              [py, str(SCRIPTS / "process_surrogates.py")],
              (PROCESSED / "surrogate-accuracy.csv",)),
        Stage("surrogate-plot", "plot surrogate accuracy versus budget",
              [py, str(SCRIPTS / "plot_surrogate_accuracy.py")],
              (FIGURES / "surrogate-accuracy-vs-budget.png",)),
        Stage("native-dags", "compile TDAR artifacts to exact CPWA-ReLU DAGs",
              [py, str(SCRIPTS / "build_native_dags.py"), "--budgets", *budget_args, *common_cpwa],
              tuple(RAW / "surrogate-performance" / f"budget-{b}" / "artifacts" / f"methane-z-b{b}-native.cpwa" for b in budgets)),
        Stage("calibrate", "calibrate every legal GPU execution policy",
              [cpwa_py, str(SCRIPTS / "calibrate_surrogate_execution.py"), "--budgets", *budget_args, "--batches", *batch_args, "--dtype", args.dtype, "--repeats", str(args.repeats), "--screen-repeats", "2", "--confirm-top", "2", "--near-tie-percent", "10", "--warmup", str(args.warmup), "--seed", str(args.calibration_seed), *common_cpwa],
              tuple(RAW / "surrogate-performance" / f"budget-{b}" / "calibration" / f"{args.dtype}-b{n}.json" for b in budgets for n in batches)),
        Stage("surrogate-benchmark", "benchmark empirically selected GPU policies",
              [cpwa_py, str(SCRIPTS / "collect_surrogate_performance.py"), "--budgets", *budget_args, "--batches", *batch_args, "--dtype", args.dtype, "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--seed", str(args.benchmark_seed), *common_cpwa],
              (RAW / "surrogate-performance" / f"benchmark-{args.dtype}.csv", RAW / "surrogate-performance" / f"benchmark-{args.dtype}.json")),
        Stage("performance-process", "join surrogate accuracy and performance",
              [py, str(SCRIPTS / "process_surrogate_performance.py")],
              (PROCESSED / "surrogate-performance.csv",)),
        Stage("performance-plot", "plot exact CPWA-ReLU throughput",
              [py, str(SCRIPTS / "plot_surrogate_crossover.py")],
              (FIGURES / "surrogate-throughput-vs-batch.png",)),
        Stage("report", "generate the reproducible V1 Markdown report",
              [py, str(SCRIPTS / "generate_report.py")],
              (REPORTS / "thermogpu-v1.md",)),
    ]


def outputs_ready(stage: Stage) -> bool:
    return bool(stage.outputs) and all(p.exists() and p.stat().st_size > 0 for p in stage.outputs)


def write_manifest(records: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), "stages": records}, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=STUDY / "configs/study-v1.toml")
    p.add_argument("--thermogpu-root", type=Path, default=Path("/nvme/Sync/ThermoGPU"))
    p.add_argument("--tdar-root", type=Path, default=Path("/nvme/Sync/tdar"))
    p.add_argument("--cpwa-relu-root", type=Path, default=Path("/nvme/Sync/cpwa-relu"))
    p.add_argument("--cpwa-python", type=Path, default=Path("/nvme/Sync/cpwa-relu/.venv/bin/python"), help="Python with the GPU-enabled JAX/CPWA-ReLU environment")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--calibration-seed", type=int, default=20261004)
    p.add_argument("--benchmark-seed", type=int, default=20261005)
    p.add_argument("--from-stage")
    p.add_argument("--through-stage")
    p.add_argument("--force", action="store_true", help="rerun selected stages even when outputs already exist")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-stages", action="store_true")
    args = p.parse_args()

    stages = build_stages(args)
    names = [s.name for s in stages]
    if args.list_stages:
        for s in stages:
            print(f"{s.name:22s} {s.description}")
        return
    for value, flag in ((args.from_stage, "--from-stage"), (args.through_stage, "--through-stage")):
        if value and value not in names:
            p.error(f"{flag}: unknown stage {value!r}; choose from {', '.join(names)}")
    lo = names.index(args.from_stage) if args.from_stage else 0
    hi = names.index(args.through_stage) if args.through_stage else len(stages) - 1
    if lo > hi:
        p.error("--from-stage occurs after --through-stage")
    selected = stages[lo:hi + 1]

    records = []
    for stage in selected:
        ready = outputs_ready(stage)
        if ready and not args.force:
            print(f"SKIP {stage.name}: outputs already present")
            records.append({"stage": stage.name, "status": "skipped", "outputs": [str(x.relative_to(ROOT)) for x in stage.outputs]})
            continue
        print(f"RUN  {stage.name}: {stage.description}")
        print("     + " + " ".join(stage.command))
        if args.dry_run:
            records.append({"stage": stage.name, "status": "dry-run", "command": stage.command})
            continue
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(stage.command, cwd=ROOT)
        record = {"stage": stage.name, "status": "ok" if result.returncode == 0 else "failed", "returncode": result.returncode, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "command": stage.command, "outputs": [str(x.relative_to(ROOT)) for x in stage.outputs]}
        records.append(record)
        write_manifest(records)
        if result.returncode != 0:
            raise SystemExit(f"stage failed: {stage.name}")
        if not outputs_ready(stage):
            raise SystemExit(f"stage completed but expected outputs are missing: {stage.name}")
    if not args.dry_run:
        write_manifest(records)


if __name__ == "__main__":
    main()
