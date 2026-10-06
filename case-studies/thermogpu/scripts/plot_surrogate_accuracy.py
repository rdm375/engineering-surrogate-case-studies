#!/usr/bin/env python3
"""Plot methane-Z surrogate error versus TDAR sample budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    ROOT
    / "case-studies/thermogpu/processed-results/surrogate-accuracy.csv"
)

DEFAULT_OUTPUT = (
    ROOT
    / "case-studies/thermogpu/figures/surrogate-accuracy-vs-budget.png"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("surrogate accuracy table is empty")

    budget = [int(r["budget"]) for r in rows]

    series = [
        ("RMSE", [float(r["rmse"]) for r in rows]),
        ("P99 absolute error", [float(r["p99_abs"]) for r in rows]),
        ("Maximum absolute error", [float(r["linf"]) for r in rows]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for label, values in series:
        ax.plot(budget, values, marker="o", label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_xticks(budget)
    ax.set_xticklabels([str(x) for x in budget])

    ax.set_xlabel("TDAR sample budget")
    ax.set_ylabel("Absolute error in compressibility factor Z")
    ax.set_title("Methane Peng–Robinson Z: Surrogate Accuracy")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(args.output)


if __name__ == "__main__":
    main()
