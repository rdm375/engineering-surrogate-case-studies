#!/usr/bin/env python3
"""Plot execution-family and global heterogeneous Pareto frontiers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
INPUT = STUDY / "processed-results/heterogeneous-pareto-candidates.csv"
OUTPUT_DIR = STUDY / "figures/heterogeneous-pareto"

COMPONENTS = range(1, 6)
BATCHES = (100, 1000, 10000, 100000, 1000000)

# Color and marker identify execution family everywhere: plot and legend.
FAMILIES = {
    "Exact CPU": {
        "field": "pareto_exact_cpu",
        "color": "C0",
        "marker": "X",
        "linestyle": "None",
    },
    "Exact CPU + OpenMP": {
        "field": "pareto_exact_cpu_omp",
        "color": "C1",
        "marker": "P",
        "linestyle": "None",
    },
    "CPWA CPU (AVX2)": {
        "field": "pareto_cpwa_cpu",
        "color": "C2",
        "marker": "o",
        "linestyle": "-",
    },
    "CPWA CPU + OpenMP": {
        "field": "pareto_cpwa_cpu_omp",
        "color": "C3",
        "marker": "s",
        "linestyle": "-",
    },
    "Exact GPU": {
        "field": "pareto_exact_gpu",
        "color": "C4",
        "marker": "D",
        "linestyle": "None",
    },
    "CPWA GPU": {
        "field": "pareto_cpwa_gpu",
        "color": "C5",
        "marker": "^",
        "linestyle": "-",
    },
}


def selected(rows, field):
    return sorted(
        [r for r in rows if r[field] == "True"],
        key=lambda r: (float(r["rmse"]), float(r["ns_per_eval"])),
    )


def batch_label(batch):
    if batch >= 1_000_000:
        return f"{batch // 1_000_000}M"
    if batch >= 1_000:
        return f"{batch // 1_000}k"
    return str(batch)


with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
for r in rows:
    groups[(int(r["components"]), int(r["batch"]))].append(r)

expected = {
    (c, batch)
    for c in COMPONENTS
    for batch in BATCHES
}
if set(groups) != expected:
    raise RuntimeError("incomplete (components, batch) coverage")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove old generated individual plots so the directory exactly reflects
# this invocation.
for old in OUTPUT_DIR.glob("C*-b*.png"):
    old.unlink()

legend_handles = []

for name, style in FAMILIES.items():
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            markersize=7,
            label=name,
        )
    )

legend_handles.append(
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=3.0,
        label="Global heterogeneous frontier",
    )
)

for c in COMPONENTS:
    for batch in BATCHES:
        candidates = groups[(c, batch)]

        fig, ax = plt.subplots(figsize=(11.0, 7.2))

        # Plot all six execution-family frontiers.  These remain visible
        # whether or not they belong to the global heterogeneous frontier.
        for name, style in FAMILIES.items():
            pts = selected(candidates, style["field"])

            xs = [float(r["rmse"]) for r in pts]
            ys = [float(r["ns_per_eval"]) for r in pts]

            if len(pts) == 1:
                ax.scatter(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    s=85,
                    zorder=4,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.7,
                    markersize=5,
                    alpha=0.82,
                    zorder=2,
                )

        # Global lower envelope.  Plot it after the family curves, but
        # slightly transparent so coincident colored family information
        # is not completely erased.
        global_pts = selected(candidates, "pareto")

        ax.plot(
            [float(r["rmse"]) for r in global_pts],
            [float(r["ns_per_eval"]) for r in global_pts],
            color="black",
            linestyle="-",
            linewidth=3.0,
            alpha=0.68,
            zorder=3,
        )

        # Put the actual globally selected points on top of the black line.
        # Their family color makes architecture switches directly visible.
        for r in global_pts:
            backend = r["backend"]

            if backend == "scalar":
                family = "Exact CPU"
            elif backend == "openmp":
                family = "Exact CPU + OpenMP"
            elif backend == "avx2":
                family = "CPWA CPU (AVX2)"
            elif backend == "avx2-omp4":
                family = "CPWA CPU + OpenMP"
            elif backend in {
                "cuda_fixed_resident",
                "cuda_generic_resident",
            }:
                family = "Exact GPU"
            elif backend == "cpwa-cuda":
                family = "CPWA GPU"
            else:
                raise RuntimeError(f"unknown backend: {backend}")

            style = FAMILIES[family]

            ax.scatter(
                [float(r["rmse"])],
                [float(r["ns_per_eval"])],
                color=style["color"],
                marker=style["marker"],
                s=72,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

        # Label budgets on each CPWA family near its corresponding point.
        # Small family-specific offsets reduce direct label collisions.
        offsets = {
            "CPWA CPU (AVX2)": (5, 7),
            "CPWA CPU + OpenMP": (5, -12),
            "CPWA GPU": (5, 7),
        }

        for name in (
            "CPWA CPU (AVX2)",
            "CPWA CPU + OpenMP",
            "CPWA GPU",
        ):
            style = FAMILIES[name]
            pts = selected(candidates, style["field"])

            for r in pts:
                dx, dy = offsets[name]
                ax.annotate(
                    f"B{r['budget']}",
                    (
                        float(r["rmse"]),
                        float(r["ns_per_eval"]),
                    ),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7,
                    color=style["color"],
                    alpha=0.9,
                )

        ax.set_yscale("log")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Execution cost (ns/eval)")
        ax.grid(True, which="both", alpha=0.20)

        ax.set_title(
            f"C{c} — batch {batch_label(batch)}\n"
            "Execution-family Pareto frontiers and heterogeneous envelope"
        )

        ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=True,
        )

        fig.tight_layout()

        output = OUTPUT_DIR / f"C{c}-b{batch}.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)

        print(output)

print(f"generated {len(expected)} figures")
