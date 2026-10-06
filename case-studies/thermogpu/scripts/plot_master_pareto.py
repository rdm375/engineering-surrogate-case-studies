#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"

INPUT = (
    STUDY
    / "processed-results/master-performance-terrain.csv"
)
OUTPUT = (
    STUDY
    / "figures/master-performance-terrain.png"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()

    with a.input.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("master terrain input is empty")

    groups = defaultdict(list)

    for r in rows:
        groups[r["method"]].append(r)

    for rs in groups.values():
        rs.sort(key=lambda r: int(r["batch_size"]))

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(15, 11),
        constrained_layout=True,
    )

    throughput_ax, latency_ax, accuracy_ax, crossover_ax = (
        axs.flat
    )

    # A/B: complete execution terrain.
    for name, rs in groups.items():
        batch = [int(r["batch_size"]) for r in rs]

        throughput_ax.plot(
            batch,
            [
                float(r["evaluations_per_second"])
                for r in rs
            ],
            marker="o",
            label=name,
        )

        latency_ax.plot(
            batch,
            [float(r["ns_per_eval"]) for r in rs],
            marker="o",
            label=name,
        )

    for ax, ylabel, title in (
        (
            throughput_ax,
            "Evaluations / second",
            "A. Throughput terrain",
        ),
        (
            latency_ax,
            "ns / evaluation",
            "B. Latency terrain",
        ),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

    # C: accuracy/performance terrain. Exact implementations stay
    # at genuine zero error; symlog avoids inventing an epsilon.
    positive_errors = [
        float(r["linf"])
        for r in rows
        if float(r["linf"]) > 0
    ]

    linthresh = (
        min(positive_errors) / 5
        if positive_errors
        else 1e-6
    )

    for name, rs in groups.items():
        accuracy_ax.scatter(
            [float(r["ns_per_eval"]) for r in rs],
            [float(r["linf"]) for r in rs],
            s=34,
            label=name,
        )

    accuracy_ax.set_xscale("log")
    accuracy_ax.set_yscale(
        "symlog",
        linthresh=linthresh,
    )
    accuracy_ax.set_xlabel("ns / evaluation")
    accuracy_ax.set_ylabel("L∞ absolute Z error")
    accuracy_ax.set_title(
        "C. Accuracy–performance terrain"
    )
    accuracy_ax.axhline(0, linewidth=0.8)
    accuracy_ax.text(
        0.02,
        0.035,
        "Exact",
        transform=accuracy_ax.transAxes,
        fontsize=9,
    )
    accuracy_ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    # D: the central engineering question.
    #
    # ratio < 1 : surrogate is faster
    # ratio = 1 : crossover
    # ratio > 1 : exact implementation is faster
    surrogate_groups = defaultdict(list)

    for r in rows:
        if r["backend"] == "gpu_surrogate":
            surrogate_groups[r["method"]].append(r)

    for name, rs in surrogate_groups.items():
        rs.sort(key=lambda r: int(r["batch_size"]))

        crossover_ax.plot(
            [int(r["batch_size"]) for r in rs],
            [
                float(r["cost_ratio_vs_best_exact"])
                for r in rs
            ],
            marker="o",
            label=name,
        )

    crossover_ax.set_xscale("log")
    crossover_ax.set_yscale("log")
    crossover_ax.set_xlabel("Batch size")
    crossover_ax.set_ylabel(
        "Surrogate cost / best exact cost"
    )
    crossover_ax.set_title(
        "D. Surrogate crossover against best exact method"
    )
    crossover_ax.axhline(1.0, linewidth=1.0)
    crossover_ax.text(
        0.02,
        0.04,
        "surrogate faster",
        transform=crossover_ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    crossover_ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    handles, labels = (
        throughput_ax.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        fontsize=9,
    )

    fig.suptitle(
        "ThermoGPU methane Z: master computation terrain\n"
        "CPU scalar · CPU parallel · direct GPU · "
        "GPU CPWA-ReLU surrogates",
        fontsize=15,
    )

    a.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(a.output, dpi=170)
    plt.close(fig)

    print(a.output)


if __name__ == "__main__":
    main()
