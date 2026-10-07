#!/usr/bin/env python3
"""Generate the publication figures for fixed-mixture EOS Case Study 2."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
PROCESSED = STUDY / "processed-results"
FIGURES = STUDY / "figures"

BATCHES = (1, 10, 100, 1000, 10000, 100000, 1000000)
BUDGETS = (16, 24, 32, 48, 64, 96, 128)
COMPONENTS = (1, 2, 3, 4, 5)


def read(name):
    with (PROCESSED / name).open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


def figure_pareto(pareto):
    """Accuracy/latency Pareto terrain at representative workloads."""
    selected_batches = (100, 1000, 1000000)

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(15.5, 9.5),
        sharex=True,
    )
    axs = axs.flat

    for i, c in enumerate(COMPONENTS):
        ax = axs[i]

        for batch in selected_batches:
            rows = [
                r for r in pareto
                if int(r["components"]) == c
                and int(r["batch"]) == batch
            ]
            rows.sort(key=lambda r: float(r["rmse"]))

            ax.plot(
                [float(r["rmse"]) for r in rows],
                [float(r["ns_per_eval"]) for r in rows],
                marker="o",
                label=f"batch {batch:,}",
            )

        ax.set_yscale("log")
        ax.set_title(f"C{c}")
        ax.set_xlabel("RMSE in compressibility factor Z")
        ax.set_ylabel("Nanoseconds per evaluation")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    axs[5].axis("off")

    fig.suptitle(
        "Fixed-composition EOS: batch-conditioned Pareto frontiers",
        fontsize=14,
    )
    save(fig, "pareto-by-mixture.png")


def figure_crossover(crossover):
    """Measured accuracy threshold at which a surrogate beats exact."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for c in COMPONENTS:
        rows = [
            r for r in crossover
            if int(r["components"]) == c
            and r["any_surrogate_faster"] == "True"
        ]
        rows.sort(key=lambda r: int(r["batch"]))

        ax.plot(
            [int(r["batch"]) for r in rows],
            [float(r["most_accurate_faster_rmse"]) for r in rows],
            marker="o",
            label=f"C{c}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size [states]")
    ax.set_ylabel("Smallest measured RMSE that beats exact")
    ax.set_title("Accuracy threshold for surrogate crossover")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    save(fig, "surrogate-crossover-threshold.png")


def figure_component_scaling(exact, surrogate):
    """Direct-vs-surrogate scaling with physical-model complexity."""
    batch = 1000000

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    best_exact = []
    for c in COMPONENTS:
        rows = [
            r for r in exact
            if int(r["components"]) == c
            and int(r["batch"]) == batch
            and r["is_best_exact"] == "True"
        ]
        assert len(rows) == 1
        best_exact.append(float(rows[0]["ns_per_eval"]))

    ax.plot(
        COMPONENTS,
        best_exact,
        marker="o",
        linewidth=2,
        label="Best exact",
    )

    for budget in (16, 64, 96, 128):
        values = []
        for c in COMPONENTS:
            rows = [
                r for r in surrogate
                if int(r["components"]) == c
                and int(r["batch"]) == batch
                and int(r["budget"]) == budget
            ]
            assert len(rows) == 1
            values.append(float(rows[0]["ns_per_eval"]))

        ax.plot(
            COMPONENTS,
            values,
            marker="o",
            label=f"Surrogate B{budget}",
        )

    ax.set_yscale("log")
    ax.set_xticks(COMPONENTS)
    ax.set_xlabel("Number of mixture components")
    ax.set_ylabel("Nanoseconds per evaluation")
    ax.set_title(
        "Evaluation cost vs physical-model complexity "
        "(batch 1,000,000)"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    save(fig, "component-scaling.png")


def figure_speedup(crossover):
    """Maximum surrogate speedup versus mixture complexity."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for batch in (100, 1000, 10000, 100000, 1000000):
        values = []
        for c in COMPONENTS:
            rows = [
                r for r in crossover
                if int(r["components"]) == c
                and int(r["batch"]) == batch
            ]
            assert len(rows) == 1
            values.append(float(rows[0]["max_speedup"]))

        ax.plot(
            COMPONENTS,
            values,
            marker="o",
            label=f"batch {batch:,}",
        )

    ax.axhline(1.0, linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(COMPONENTS)
    ax.set_xlabel("Number of mixture components")
    ax.set_ylabel("Maximum measured speedup vs best exact")
    ax.set_title("Surrogate economic value vs mixture complexity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    save(fig, "speedup-vs-components.png")


def figure_resources(resources):
    """Show register saturation and spill growth with surrogate budget."""
    fig, (ax_reg, ax_spill) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.0),
    )

    for c in COMPONENTS:
        rows = [
            r for r in resources
            if int(r["components"]) == c
        ]
        rows.sort(key=lambda r: int(r["budget"]))

        budgets = [int(r["budget"]) for r in rows]

        ax_reg.plot(
            budgets,
            [int(r["registers"]) for r in rows],
            marker="o",
            label=f"C{c}",
        )

        ax_spill.plot(
            budgets,
            [int(r["spill_store_bytes"]) for r in rows],
            marker="o",
            label=f"C{c}",
        )

    ax_reg.set_xlabel("TDAR sample budget")
    ax_reg.set_ylabel("Registers")
    ax_reg.set_title("Register pressure")
    ax_reg.grid(True, alpha=0.25)
    ax_reg.legend()

    ax_spill.set_xlabel("TDAR sample budget")
    ax_spill.set_ylabel("Spill-store bytes")
    ax_spill.set_title("Local-memory spilling")
    ax_spill.grid(True, alpha=0.25)
    ax_spill.legend()

    fig.suptitle(
        "Hardware resource pressure vs approximation complexity",
        fontsize=14,
    )

    save(fig, "resource-pressure.png")


def figure_accuracy(resources):
    """Approximation accuracy as TDAR budget increases."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for c in COMPONENTS:
        rows = [
            r for r in resources
            if int(r["components"]) == c
        ]
        rows.sort(key=lambda r: int(r["budget"]))

        ax.plot(
            [int(r["budget"]) for r in rows],
            [float(r["rmse"]) for r in rows],
            marker="o",
            label=f"C{c}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("TDAR sample budget")
    ax.set_ylabel("RMSE in compressibility factor Z")
    ax.set_title("Surrogate accuracy vs adaptive-sampling budget")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    save(fig, "accuracy-vs-budget.png")


def main():
    exact = read("exact-performance.csv")
    surrogate = read("surrogate-performance.csv")
    pareto = read("pareto-frontiers.csv")
    crossover = read("surrogate-crossover.csv")
    resources = read("resource-pressure.csv")

    assert len(exact) == 245
    assert len(surrogate) == 245
    assert len(crossover) == 35
    assert len(resources) == 35

    figure_pareto(pareto)
    figure_crossover(crossover)
    figure_component_scaling(exact, surrogate)
    figure_speedup(crossover)
    figure_resources(resources)
    figure_accuracy(resources)

    print("figures generated: 6")


if __name__ == "__main__":
    main()
