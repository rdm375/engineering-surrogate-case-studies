#!/usr/bin/env python3
"""Plot winning execution architecture for the post-v1 heterogeneous study."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

INPUT = STUDY / "processed-results/heterogeneous-execution.csv"
OUTPUT = STUDY / "figures/execution-regimes.png"

BATCHES = [100, 1000, 10000, 100000, 1000000]
TARGETS = [1e-3, 7.5e-4, 5e-4, 4e-4]

ARCH = {
    "cpwa-cpu-avx2": 0,
    "cpwa-cpu-avx2-omp4": 1,
    "cpwa-cuda": 2,
}

LABEL = {
    0: "AVX2",
    1: "AVX2+OMP4",
    2: "CUDA",
}


with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

lookup = {
    (
        int(r["components"]),
        float(r["rmse_target"]),
        int(r["batch"]),
    ): r["winner"]
    for r in rows
}

# Deliberately use the default matplotlib qualitative palette.
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cmap = ListedColormap(colors[:3])

fig, axes = plt.subplots(
    1, 5,
    figsize=(15, 4.4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for components, ax in enumerate(axes, start=1):
    z = np.empty((len(TARGETS), len(BATCHES)), dtype=int)

    for yi, target in enumerate(TARGETS):
        for xi, batch in enumerate(BATCHES):
            winner = lookup[(components, target, batch)]
            if winner not in ARCH:
                raise RuntimeError(
                    f"unexpected winner C{components}, "
                    f"target={target}, batch={batch}: {winner}"
                )
            z[yi, xi] = ARCH[winner]

    ax.imshow(
        z,
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
    )

    for yi in range(len(TARGETS)):
        for xi in range(len(BATCHES)):
            ax.text(
                xi,
                yi,
                LABEL[z[yi, xi]],
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title(f"C{components}")
    ax.set_xticks(range(len(BATCHES)))
    ax.set_xticklabels(
        ["100", "1k", "10k", "100k", "1M"],
        rotation=45,
        ha="right",
    )

axes[0].set_yticks(range(len(TARGETS)))
axes[0].set_yticklabels(
    [r"$10^{-3}$", r"$7.5\times10^{-4}$",
     r"$5\times10^{-4}$", r"$4\times10^{-4}$"]
)

fig.supxlabel("Batch size")
fig.supylabel("RMSE requirement")
fig.suptitle(
    "Winning CPWA execution architecture by operating regime"
)

legend = [
    Patch(facecolor=colors[0], label="AVX2"),
    Patch(facecolor=colors[1], label="AVX2 + OpenMP 4"),
    Patch(facecolor=colors[2], label="Fused CUDA"),
]
fig.legend(
    handles=legend,
    loc="outside lower center",
    ncol=3,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180)
plt.close(fig)

print(OUTPUT)
