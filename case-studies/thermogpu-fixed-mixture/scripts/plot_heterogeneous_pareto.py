#!/usr/bin/env python3
"""Plot all heterogeneous Pareto frontiers for Case Study 2."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

INPUT = STUDY / "processed-results/heterogeneous-pareto-frontiers.csv"
OUTPUT = STUDY / "figures/heterogeneous-pareto-frontiers.png"

COMPONENTS = (1, 2, 3, 4, 5)
BATCHES = (100, 1000, 10000, 100000, 1000000)

MARKERS = {
    "exact-cpu": "X",
    "exact-cuda": "D",
    "avx2": "o",
    "avx2-omp4": "s",
    "cpwa-cuda": "^",
}

LABELS = {
    "exact-cpu": "Exact CPU",
    "exact-cuda": "Exact CUDA",
    "avx2": "CPWA AVX2",
    "avx2-omp4": "CPWA AVX2 + OpenMP 4",
    "cpwa-cuda": "CPWA fused CUDA",
}

COLORS = {
    "exact-cpu": "C0",
    "exact-cuda": "C1",
    "avx2": "C2",
    "avx2-omp4": "C3",
    "cpwa-cuda": "C4",
}


def architecture(row: dict[str, str]) -> str:
    if row["candidate_type"] == "exact":
        if row["backend"].startswith("cuda"):
            return "exact-cuda"
        return "exact-cpu"

    return row["backend"]


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
    raise RuntimeError("incomplete Pareto group coverage")

fig, axes = plt.subplots(
    5,
    5,
    figsize=(15.5, 13.0),
    sharex=True,
    sharey=True,
)

# Reserve a dedicated right-hand strip for the legend.
fig.subplots_adjust(
    left=0.08,
    right=0.82,
    bottom=0.08,
    top=0.92,
    wspace=0.16,
    hspace=0.18,
)

for row_i, c in enumerate(COMPONENTS):
    for col_i, batch in enumerate(BATCHES):
        ax = axes[row_i, col_i]

        points = sorted(
            groups[(c, batch)],
            key=lambda r: float(r["rmse"]),
        )

        xs = [float(r["rmse"]) for r in points]
        ys = [float(r["ns_per_eval"]) for r in points]

        # Neutral line represents the actual nondominated tradeoff.
        ax.plot(
            xs,
            ys,
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )

        for kind in MARKERS:
            selected = [
                r for r in points
                if architecture(r) == kind
            ]

            if not selected:
                continue

            ax.scatter(
                [float(r["rmse"]) for r in selected],
                [float(r["ns_per_eval"]) for r in selected],
                marker=MARKERS[kind],
                color=COLORS[kind],
                s=30,
                zorder=2,
            )

        # Label the budget progression once as a visual key.
        if c == 5 and batch == 1_000_000:
            for r in points:
                if r["candidate_type"] != "surrogate":
                    continue

                # Only label the surviving CUDA representation of each
                # mathematical surrogate in this panel.
                if architecture(r) != "cpwa-cuda":
                    continue

                ax.annotate(
                    f"B{r['budget']}",
                    (
                        float(r["rmse"]),
                        float(r["ns_per_eval"]),
                    ),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                )

        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.20)

        if row_i == 0:
            if batch >= 1_000_000:
                title = "1M"
            elif batch >= 1_000:
                title = f"{batch // 1000}k"
            else:
                title = str(batch)
            ax.set_title(f"batch {title}")

        if col_i == 0:
            ax.set_ylabel(f"C{c}\nns/eval")

        if row_i == len(COMPONENTS) - 1:
            ax.set_xlabel("RMSE")

fig.suptitle(
    "Heterogeneous Pareto frontiers: exact physics and CPWA execution",
    fontsize=14,
)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker=MARKERS[kind],
        markerfacecolor=COLORS[kind],
        markeredgecolor=COLORS[kind],
        linestyle="None",
        markersize=7,
        label=LABELS[kind],
    )
    for kind in (
        "exact-cpu",
        "exact-cuda",
        "avx2",
        "avx2-omp4",
        "cpwa-cuda",
    )
]

fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.84, 0.52),
    frameon=False,
    title="Pareto-optimal backend",
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180)
plt.close(fig)

print(OUTPUT)
