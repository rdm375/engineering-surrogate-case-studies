#!/usr/bin/env python3
"""Collect direct ThermoGPU scaling measurements.

ThermoGPU owns the direct-model benchmark methodology.  This collector invokes
the released ``thermogpu_scaling`` executable, retains its native output, and
normalizes every CSV measurement row into JSON.

No backend or thread-count selection is performed here.  In particular, the
"best parallel CPU" configuration is a derived reporting result, not a
collection-time choice.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from engineering_case_studies.collection import (
    run_capture,
    slurm_metadata,
    write_json,
)
from engineering_case_studies.provenance import git_state


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "case-studies/thermogpu/raw-results/direct"
STUDY = "thermogpu-methane-z-v1"

CSV_HEADER = [
    "components",
    "states",
    "backend",
    "threads",
    "calls_per_sample",
    "median_seconds",
    "ns_per_state",
    "states_per_second",
    "mad_percent",
    "speedup_vs_scalar",
    "speedup_vs_best_cpu",
]

INT_FIELDS = {
    "components",
    "states",
    "threads",
    "calls_per_sample",
}

FLOAT_FIELDS = {
    "median_seconds",
    "ns_per_state",
    "states_per_second",
    "mad_percent",
    "speedup_vs_scalar",
    "speedup_vs_best_cpu",
}


def parse_scaling_output(text: str) -> tuple[str, list[dict[str, object]]]:
    """Parse ThermoGPU scaling output without discarding native metadata."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    try:
        header_index = next(
            i for i, line in enumerate(lines)
            if line.startswith("components,states,backend,threads,")
        )
    except StopIteration as exc:
        raise ValueError("ThermoGPU scaling CSV header not found") from exc

    preamble = "\n".join(lines[:header_index])
    csv_text = "\n".join(lines[header_index:]) + "\n"

    reader = csv.DictReader(io.StringIO(csv_text))

    if reader.fieldnames != CSV_HEADER:
        raise ValueError(
            "unexpected ThermoGPU scaling CSV schema: "
            f"{reader.fieldnames!r}"
        )

    rows: list[dict[str, object]] = []

    for raw in reader:
        row: dict[str, object] = {}

        for key, value in raw.items():
            if value is None:
                raise ValueError(f"missing value for field {key!r}")

            if key in INT_FIELDS:
                row[key] = int(value)
            elif key in FLOAT_FIELDS:
                row[key] = float(value)
            else:
                row[key] = value

        if int(row["components"]) <= 0:
            raise ValueError("components must be positive")
        if int(row["states"]) <= 0:
            raise ValueError("states must be positive")
        if float(row["ns_per_state"]) <= 0:
            raise ValueError("ns_per_state must be positive")
        if float(row["states_per_second"]) <= 0:
            raise ValueError("states_per_second must be positive")

        rows.append(row)

    if not rows:
        raise ValueError("ThermoGPU scaling output contained no measurements")

    return preamble, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--thermogpu-root", type=Path, required=True)
    p.add_argument(
        "--batch",
        type=int,
        nargs="+",
        required=True,
        help="one or more ThermoGPU state counts",
    )
    p.add_argument("--executable", type=Path)
    args = p.parse_args()

    batches = args.batch

    if any(batch <= 0 for batch in batches):
        raise SystemExit("all batch sizes must be positive")

    exe = args.executable or args.thermogpu_root / "build/thermogpu_scaling"

    if not exe.exists():
        raise SystemExit(f"ThermoGPU scaling executable not found: {exe}")

    cmd = [str(exe), *map(str, batches)]
    run = run_capture(cmd, cwd=args.thermogpu_root)

    batch_label = "-".join(map(str, batches))
    stem = f"scaling-b{batch_label}"

    RAW.mkdir(parents=True, exist_ok=True)

    stdout_path = RAW / f"{stem}.stdout.txt"
    stderr_path = RAW / f"{stem}.stderr.txt"
    json_path = RAW / f"{stem}.json"

    stdout_path.write_text(run["stdout"])
    stderr_path.write_text(run["stderr"])

    common = {
        "study": STUDY,
        "kind": "direct-performance",
        "source": "ThermoGPU",
        "requested_batches": batches,
        "wall_seconds": run["elapsed_seconds"],
        "thermogpu": git_state(args.thermogpu_root),
        "slurm": slurm_metadata(),
        "command": run["command"],
    }

    if run["returncode"] != 0:
        write_json(
            json_path,
            {
                "status": "failed",
                **common,
                "returncode": run["returncode"],
            },
        )
        raise SystemExit(
            f"ThermoGPU scaling benchmark failed ({run['returncode']}); "
            f"native output retained in {RAW}"
        )

    try:
        preamble, rows = parse_scaling_output(run["stdout"])
    except (ValueError, TypeError) as exc:
        write_json(
            json_path,
            {
                "status": "unparsed",
                **common,
                "returncode": run["returncode"],
                "parse_error": str(exc),
            },
        )
        raise SystemExit(
            f"{exc}; native output retained in {RAW}"
        ) from exc

    observed_batches = sorted({int(row["states"]) for row in rows})
    missing_batches = sorted(set(batches) - set(observed_batches))

    record = {
        "status": "ok",
        **common,
        "benchmark_preamble": preamble,
        "observed_batches": observed_batches,
        "measurement_count": len(rows),
        "measurements": rows,
    }

    if missing_batches:
        record["missing_requested_batches"] = missing_batches

    write_json(json_path, record)

    print(
        json.dumps(
            {
                "output": str(json_path),
                "requested_batches": batches,
                "observed_batches": observed_batches,
                "measurement_count": len(rows),
                "missing_requested_batches": missing_batches,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
