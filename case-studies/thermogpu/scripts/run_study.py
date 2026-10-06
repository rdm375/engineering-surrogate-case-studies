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
REPRODUCTION_MANIFEST = RAW / "pipeline-reproduction-manifest.json"


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: list[str]
    outputs: tuple[Path, ...]
    evidence_producer: bool = False
    reproduce: bool = True


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
              (RAW / "environment.json",),
              evidence_producer=True),
        Stage("direct", "collect canonical ThermoGPU direct scaling evidence",
              [py, str(SCRIPTS / "collect_direct.py"), "--thermogpu-root", str(args.thermogpu_root.resolve()), "--batch", *batch_args],
              (RAW / "direct" / f"{direct_stem}.json",),
              evidence_producer=True),
        Stage("direct-process", "normalize direct methane results",
              [py, str(SCRIPTS / "process_direct.py")],
              (PROCESSED / "direct-methane.csv",)),
        Stage("direct-plot", "plot direct throughput scaling",
              [py, str(SCRIPTS / "plot_direct.py")],
              (FIGURES / "direct-throughput-vs-batch.png",)),
        Stage("surrogates", "build and validate the frozen TDAR surrogate sweep",
              [py, str(SCRIPTS / "build_surrogates.py"), "--budgets", *budget_args, "--initial", str(initial), "--validation-points", str(validation_points), "--seed", str(seed), "--tdar-root", str(args.tdar_root.resolve()), "--thermogpu-root", str(args.thermogpu_root.resolve())],
              (RAW / "surrogates" / "normalized.json", RAW / "surrogates" / "validation-points.npy", RAW / "surrogates" / "validation-truth.npy"),
              evidence_producer=True),
        Stage("surrogate-process", "normalize TDAR accuracy results",
              [py, str(SCRIPTS / "process_surrogates.py")],
              (PROCESSED / "surrogate-accuracy.csv",)),
        Stage("surrogate-plot", "plot surrogate accuracy versus budget",
              [py, str(SCRIPTS / "plot_surrogate_accuracy.py")],
              (FIGURES / "surrogate-accuracy-vs-budget.png",)),
        Stage("native-dags", "compile TDAR artifacts to exact CPWA-ReLU DAGs",
              [py, str(SCRIPTS / "build_native_dags.py"), "--budgets", *budget_args, *common_cpwa],
              tuple(RAW / "surrogate-performance" / f"budget-{b}" / "artifacts" / f"methane-z-b{b}-native.cpwa" for b in budgets),
              evidence_producer=True),
        Stage("calibrate", "calibrate every legal GPU execution policy",
              [cpwa_py, str(SCRIPTS / "calibrate_surrogate_execution.py"), "--budgets", *budget_args, "--batches", *batch_args, "--dtype", args.dtype, "--repeats", str(args.repeats), "--screen-repeats", "2", "--confirm-top", "2", "--near-tie-percent", "10", "--warmup", str(args.warmup), "--seed", str(args.calibration_seed), *common_cpwa],
              tuple(RAW / "surrogate-performance" / f"budget-{b}" / "calibration" / f"{args.dtype}-b{n}.json" for b in budgets for n in batches),
              evidence_producer=True),
        Stage("surrogate-benchmark", "benchmark empirically selected GPU policies",
              [cpwa_py, str(SCRIPTS / "collect_surrogate_performance.py"), "--budgets", *budget_args, "--batches", *batch_args, "--dtype", args.dtype, "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--seed", str(args.benchmark_seed), *common_cpwa],
              (RAW / "surrogate-performance" / f"benchmark-{args.dtype}.csv", RAW / "surrogate-performance" / f"benchmark-{args.dtype}.json"),
              evidence_producer=True),
        Stage("performance-process", "join legacy surrogate accuracy and performance",
              [py, str(SCRIPTS / "process_surrogate_performance.py")],
              (PROCESSED / "surrogate-performance.csv",),
              reproduce=False),
        Stage("performance-plot", "plot legacy exact CPWA-ReLU throughput",
              [py, str(SCRIPTS / "plot_surrogate_crossover.py")],
              (FIGURES / "surrogate-throughput-vs-batch.png",),
              reproduce=False),
        Stage("master-pareto", "construct the unified exact/surrogate performance terrain",
              [py, str(SCRIPTS / "process_master_pareto.py")],
              (PROCESSED / "master-performance-terrain.csv", PROCESSED / "master-pareto-global.csv", PROCESSED / "surrogate-pareto.csv")),
        Stage("analysis", "derive crossover, accuracy-cost, dominance, DAG-scaling, and feasibility results",
              [py, str(SCRIPTS / "analyze_study.py")],
              (PROCESSED / "exact-crossover.csv", PROCESSED / "surrogate-accuracy-cost.csv", PROCESSED / "dominance-analysis.csv", PROCESSED / "dag-scaling.csv", PROCESSED / "scaling-analysis.csv", PROCESSED / "feasibility-analysis.csv", PROCESSED / "study-summary.json")),
        Stage("report", "generate the reproducible V1 Markdown report",
              [py, str(SCRIPTS / "generate_report.py")],
              (REPORTS / "thermogpu-v1.md",)),
    ]


def outputs_ready(stage: Stage) -> bool:
    return bool(stage.outputs) and all(p.exists() and p.stat().st_size > 0 for p in stage.outputs)


V1_DEFERRED_CALIBRATIONS = (
    (256, 10_000),
    (256, 100_000),
    (256, 1_000_000),
)


def retained_evidence(stage: Stage) -> tuple[list[Path], list[str]]:
    """Return actual retained evidence and declared non-measurements."""
    if stage.name == "calibrate":
        evidence: list[Path] = []
        deferred = set(V1_DEFERRED_CALIBRATIONS)

        for nominal in stage.outputs:
            # .../budget-256/calibration/float32-b10000.json
            budget = int(nominal.parents[1].name.removeprefix("budget-"))
            batch = int(nominal.stem.split("-b", 1)[1])

            if (budget, batch) in deferred:
                continue

            if nominal.exists() and nominal.stat().st_size > 0:
                evidence.append(nominal)
                continue

            failure = nominal.with_suffix(".failed.json")
            if failure.exists() and failure.stat().st_size > 0:
                evidence.append(failure)

        declarations = [
            f"budget={budget},batch={batch}:deferred"
            for budget, batch in V1_DEFERRED_CALIBRATIONS
        ]
        return evidence, declarations

    if stage.name == "surrogate-benchmark":
        # V1 performance analysis consumes the retained calibration artifacts
        # directly. The historical benchmark-float32 aggregate is obsolete.
        return [], [
            "legacy aggregate not required; authoritative performance evidence "
            "is retained by calibrate"
        ]

    return [
        p for p in stage.outputs
        if p.exists() and p.stat().st_size > 0
    ], []


def retained_evidence_ready(
    stage: Stage,
) -> tuple[bool, list[str], list[Path], list[str]]:
    """Validate actual retained evidence needed by reproduction mode."""
    evidence, declarations = retained_evidence(stage)

    if stage.name == "calibrate":
        deferred = set(V1_DEFERRED_CALIBRATIONS)
        required = len(stage.outputs) - len(deferred)
        if len(evidence) == required:
            return True, [], evidence, declarations

        represented = set()
        for p in evidence:
            budget = int(p.parents[1].name.removeprefix("budget-"))
            stem = p.name.removesuffix(".failed.json").removesuffix(".json")
            batch = int(stem.split("-b", 1)[1])
            represented.add((budget, batch))

        missing = []
        for nominal in stage.outputs:
            budget = int(nominal.parents[1].name.removeprefix("budget-"))
            batch = int(nominal.stem.split("-b", 1)[1])
            cell = (budget, batch)
            if cell not in deferred and cell not in represented:
                missing.append(f"budget={budget},batch={batch}")
        return False, missing, evidence, declarations

    if stage.name == "surrogate-benchmark":
        return True, [], evidence, declarations

    missing = [
        str(p.relative_to(ROOT))
        for p in stage.outputs
        if not p.exists() or p.stat().st_size == 0
    ]
    return not missing, missing, evidence, declarations


def write_manifest(records: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "stages": records,
            },
            indent=2,
        )
        + "\n"
    )


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
    p.add_argument(
        "--reproduce",
        action="store_true",
        help=(
            "rebuild all derived results, figures, analyses, and report from "
            "retained evidence without rerunning evidence-producing stages"
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-stages", action="store_true")
    args = p.parse_args()

    if args.reproduce and args.force:
        p.error("--reproduce and --force are mutually exclusive")

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

        if args.reproduce and not stage.reproduce:
            print(f"OMIT {stage.name}: legacy stage not required for reproduction")
            records.append({
                "stage": stage.name,
                "status": "omitted-from-reproduction",
            })
            continue

        if args.reproduce and stage.evidence_producer:
            evidence_ready, missing, evidence, declarations = (
                retained_evidence_ready(stage)
            )
            if not evidence_ready:
                raise SystemExit(
                    f"cannot reproduce: retained evidence is missing for "
                    f"{stage.name}: " + ", ".join(missing)
                )
            print(f"KEEP {stage.name}: retained experimental evidence")
            record = {
                "stage": stage.name,
                "status": "retained-evidence",
                "evidence": [str(x.relative_to(ROOT)) for x in evidence],
            }
            if declarations:
                record["declarations"] = declarations
            records.append(record)
            continue

        if ready and not args.force and not args.reproduce:
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
        write_manifest(
            records,
            REPRODUCTION_MANIFEST if args.reproduce else MANIFEST,
        )
        if result.returncode != 0:
            raise SystemExit(f"stage failed: {stage.name}")
        if not outputs_ready(stage):
            raise SystemExit(f"stage completed but expected outputs are missing: {stage.name}")
    if not args.dry_run:
        write_manifest(
            records,
            REPRODUCTION_MANIFEST if args.reproduce else MANIFEST,
        )


if __name__ == "__main__":
    main()
