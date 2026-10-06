#!/usr/bin/env python3
"""Submit ThermoGPU case-study stages to Slurm.

Site-specific account/partition/QOS options are intentionally supplied at
submission time through ``--sbatch-arg`` rather than frozen in the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engineering_case_studies.executors import SlurmExecutor


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "case-studies/thermogpu/slurm"
RAW = ROOT / "case-studies/thermogpu/raw-results"

STAGES = {
    "direct-cpu": SLURM / "direct-cpu.sbatch",
    "direct-gpu": SLURM / "direct-gpu.sbatch",
    "surrogate-build": SLURM / "surrogate-build.sbatch",
    "surrogate-benchmark": SLURM / "surrogate-benchmark.sbatch",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--thermogpu-root", type=Path, default=Path("/nvme/Sync/ThermoGPU"))
    parser.add_argument("--tdar-root", type=Path, default=Path("/nvme/Sync/tdar"))
    parser.add_argument("--cpwa-relu-root", type=Path, default=Path("/nvme/Sync/cpwa-relu"))
    parser.add_argument("--cpwa-benchmark-command", default=None)
    parser.add_argument("--sbatch-arg", action="append", default=[])
    args = parser.parse_args()

    (RAW / "slurm").mkdir(parents=True, exist_ok=True)
    executor = SlurmExecutor()
    if not executor.available():
        raise SystemExit("sbatch not found; use local execution or run on a Slurm host")

    submission = executor.submit(
        STAGES[args.stage],
        exports={
            "CASE_STUDY_ROOT": str(ROOT),
            "THERMOGPU_ROOT": str(args.thermogpu_root.resolve()),
            "TDAR_ROOT": str(args.tdar_root.resolve()),
            "CPWA_RELU_ROOT": str(args.cpwa_relu_root.resolve()),
            **({"CPWA_RELU_BENCHMARK_COMMAND": args.cpwa_benchmark_command} if args.cpwa_benchmark_command else {}),
        },
        extra_sbatch_args=args.sbatch_arg,
        cwd=ROOT,
    )
    record = {
        "stage": args.stage,
        "job_id": submission.job_id,
        "script": str(submission.script.relative_to(ROOT)),
        "command": list(submission.command),
    }
    output = RAW / "slurm" / f"submission-{submission.job_id}.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
