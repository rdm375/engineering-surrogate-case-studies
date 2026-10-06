#!/usr/bin/env python3
"""Plot methane direct-model throughput versus batch size."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    ROOT
    / "case-studies/thermogpu/processed-results/direct-methane.csv"
)

DEFAULT_OUTPUT = (
    ROOT
    / "case-studies/thermogpu/figures/direct-throughput-vs-batch.png"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("processed direct-performance table is empty")

    batch = [int(r["batch"]) for r in rows]

    series = [
        (
            "Scalar CPU",
            [float(r["scalar_rate"]) for r in rows],
        ),
        (
            "Best parallel CPU",
            [float(r["best_openmp_rate"]) for r in rows],
        ),
        (
            "CUDA resident",
            [float(r["cuda_resident_rate"]) for r in rows],
        ),
        (
            "CUDA end-to-end",
            [float(r["cuda_e2e_rate"]) for r in rows],
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for label, rate in series:
        ax.plot(batch, rate, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Batch size [states]")
    ax.set_ylabel("Throughput [states/s]")
    ax.set_title("Methane Peng–Robinson Z: Direct Evaluation Throughput")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(args.output)


if __name__ == "__main__":
    main()
