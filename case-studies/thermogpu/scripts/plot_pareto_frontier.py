#!/usr/bin/env python3
"""Plot the complete ThermoGPU performance/accuracy Pareto terrain."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu"

INPUT = STUDY / "processed-results/master-performance-terrain.csv"
OUTPUT = STUDY / "figures/pareto-frontier.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()

    rows = read_rows(args.input)

    if not rows:
        raise SystemExit("master performance terrain is empty")

    rows = [r for r in rows if truth(r["feasible"])]

    exact = [r for r in rows if truth(r["exact"])]
    surrogate = [r for r in rows if not truth(r["exact"])]

    # ------------------------------------------------------------------
    # Group performance-scaling trajectories.
    # ------------------------------------------------------------------

    scaling = defaultdict(list)

    for r in rows:
        scaling[r["method"]].append(r)

    for rs in scaling.values():
        rs.sort(key=lambda r: int(r["batch_size"]))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, (ax_scale, ax_pareto) = plt.subplots(
        1,
        2,
        figsize=(16, 7.5),
        constrained_layout=True,
    )

    # ================================================================
    # Panel A: complete performance scaling terrain
    # ================================================================

    for method, rs in sorted(scaling.items()):
        x = [int(r["batch_size"]) for r in rs]
        y = [float(r["ns_per_eval"]) for r in rs]

        is_surrogate = rs[0]["backend"] == "gpu_surrogate"

        ax_scale.plot(
            x,
            y,
            marker="o",
            linewidth=1.35 if is_surrogate else 2.0,
            markersize=4.5 if is_surrogate else 6.0,
            alpha=0.72 if is_surrogate else 0.95,
            label=method,
        )

    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")

    ax_scale.set_xlabel("Batch size")
    ax_scale.set_ylabel("Nanoseconds per evaluation")
    ax_scale.set_title("A. Performance scaling")

    ax_scale.grid(True, which="both", alpha=0.25)

    # ================================================================
    # Panel B: complete accuracy/performance terrain
    # ================================================================

    positive_errors = [
        float(r["linf"])
        for r in surrogate
        if float(r["linf"]) > 0.0
    ]

    if not positive_errors:
        raise SystemExit("no positive surrogate errors found")

    linthresh = min(positive_errors) / 5.0

    # All exact measurements.
    exact_groups = defaultdict(list)

    for r in exact:
        exact_groups[r["method"]].append(r)

    for method, rs in sorted(exact_groups.items()):
        rs.sort(key=lambda r: float(r["ns_per_eval"]))

        ax_pareto.scatter(
            [float(r["ns_per_eval"]) for r in rs],
            [0.0 for _ in rs],
            s=45,
            alpha=0.55,
            label=method,
        )

    # All surrogate measurements.
    surrogate_groups = defaultdict(list)

    for r in surrogate:
        surrogate_groups[int(r["budget"])].append(r)

    for budget, rs in sorted(surrogate_groups.items()):
        rs.sort(key=lambda r: int(r["batch_size"]))

        ax_pareto.plot(
            [float(r["ns_per_eval"]) for r in rs],
            [float(r["linf"]) for r in rs],
            marker="o",
            linewidth=1.2,
            markersize=4.5,
            alpha=0.55,
            label=f"GPU surrogate b{budget}",
        )

    # ---------------------------------------------------------------
    # Surrogate-only Pareto points.
    # ---------------------------------------------------------------

    surrogate_frontier = [
        r for r in surrogate
        if truth(r["surrogate_pareto"])
    ]

    ax_pareto.scatter(
        [float(r["ns_per_eval"]) for r in surrogate_frontier],
        [float(r["linf"]) for r in surrogate_frontier],
        s=78,
        facecolors="none",
        linewidths=1.4,
        label="Surrogate frontier",
    )

    # ---------------------------------------------------------------
    # Global Pareto frontier.
    #
    # Exact points all have zero approximation error. Connecting the
    # globally nondominated exact measurements therefore shows the
    # lower performance boundary without inventing a positive error.
    # ---------------------------------------------------------------

    global_frontier = [
        r for r in rows
        if truth(r["global_pareto"])
    ]

    global_frontier.sort(key=lambda r: int(r["batch_size"]))

    ax_pareto.scatter(
        [float(r["ns_per_eval"]) for r in global_frontier],
        [float(r["linf"]) for r in global_frontier],
        marker="*",
        s=175,
        linewidths=1.1,
        label="Global Pareto frontier",
        zorder=10,
    )

    # Label the global frontier so the CPU -> OpenMP -> GPU transition
    # is explicit.
    for r in global_frontier:
        batch = int(r["batch_size"])

        ax_pareto.annotate(
            f"{r['method']}\nn={batch:,}",
            (
                float(r["ns_per_eval"]),
                float(r["linf"]),
            ),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax_pareto.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
    )

    ax_pareto.set_xscale("log")
    ax_pareto.set_yscale(
        "symlog",
        linthresh=linthresh,
    )

    ax_pareto.set_xlabel("Nanoseconds per evaluation")
    ax_pareto.set_ylabel(
        r"$L_\infty$ absolute error in compressibility factor $Z$"
    )
    ax_pareto.set_title("B. Accuracy–performance Pareto terrain")

    ax_pareto.grid(True, which="both", alpha=0.25)

    ax_pareto.text(
        0.02,
        0.035,
        "Exact computation",
        transform=ax_pareto.transAxes,
        fontsize=9,
    )

    # ------------------------------------------------------------------
    # Figure-level legend.
    #
    # De-duplicate labels because exact methods occur in both panels.
    # ------------------------------------------------------------------

    handles = []
    labels = []
    seen = set()

    for ax in (ax_scale, ax_pareto):
        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(h, l):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        fontsize=8.5,
    )

    fig.suptitle(
        "ThermoGPU methane Z: complete computation Pareto frontier\n"
        "CPU scalar · CPU parallel · direct GPU · GPU CPWA-ReLU surrogates",
        fontsize=15,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(args.output)
    print(
        f"records={len(rows)} "
        f"exact={len(exact)} "
        f"surrogate={len(surrogate)} "
        f"global_frontier={len(global_frontier)} "
        f"surrogate_frontier={len(surrogate_frontier)}"
    )


if __name__ == "__main__":
    main()
