#!/usr/bin/env python3
"""Plot B96 -> B128 DAG and execution-cost growth by backend."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

INPUT = STUDY / "processed-results/high-budget-backend-scaling.csv"
OUTPUT = STUDY / "figures/high-budget-backend-scaling.png"


with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 5:
    raise RuntimeError(f"expected 5 scaling rows, found {len(rows)}")

components = [f"C{r['components']}" for r in rows]

node_growth = np.array([float(r["node_growth"]) for r in rows])
avx2_growth = np.array([float(r["avx2_growth"]) for r in rows])
omp4_growth = np.array([float(r["omp4_growth"]) for r in rows])
cuda_growth = np.array([float(r["cuda_growth"]) for r in rows])

x = np.arange(len(rows))
width = 0.19

fig, ax = plt.subplots(figsize=(10.5, 5.8))

bars_nodes = ax.bar(
    x - 1.5 * width,
    node_growth,
    width,
    label="DAG nodes",
)
bars_avx2 = ax.bar(
    x - 0.5 * width,
    avx2_growth,
    width,
    label="AVX2",
)
bars_omp4 = ax.bar(
    x + 0.5 * width,
    omp4_growth,
    width,
    label="AVX2 + OpenMP 4",
)
bars_cuda = ax.bar(
    x + 1.5 * width,
    cuda_growth,
    width,
    label="Fused CUDA",
)

ax.axhline(
    1.0,
    linewidth=1.0,
    linestyle="--",
)

ax.set_xticks(x)
ax.set_xticklabels(components)
ax.set_ylabel("B128 / B96 ratio")
ax.set_xlabel("Fixed-composition mixture")
ax.set_title(
    "High-budget execution growth versus realized DAG growth\n"
    "Batch 1,000,000"
)

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    frameon=False,
    ncol=2,
)

# Label CUDA bars because their departure from DAG growth is the
# principal feature of this figure.
for bar, value in zip(bars_cuda, cuda_growth):
    ax.annotate(
        f"{value:.2f}×",
        (
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
        ),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180)
plt.close(fig)

print(OUTPUT)
