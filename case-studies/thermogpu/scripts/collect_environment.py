#!/usr/bin/env python3
"""Capture machine and frozen-repository provenance for the ThermoGPU study."""

from __future__ import annotations

import argparse
from pathlib import Path

from engineering_case_studies.provenance import save_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/nvme/Sync"),
        help="Directory containing the four study repositories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the environment manifest output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    case_repo = workspace / "engineering-surrogate-case-studies"
    output = args.output or (
        case_repo / "case-studies/thermogpu/raw-results/environment.json"
    )
    save_environment(
        output,
        {
            "ThermoGPU": workspace / "ThermoGPU",
            "TDAR": workspace / "tdar",
            "CPWA-ReLU": workspace / "cpwa-relu",
            "case-study": case_repo,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
