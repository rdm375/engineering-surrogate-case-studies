#!/usr/bin/env python3
"""Compare native CPU and fused CUDA scaling with realized CPWA DAG size."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"

CPU_INPUT = STUDY / "evidence/native-cpu-surrogate/summary.csv"
GPU_INPUT = STUDY / "processed-results/surrogate-performance.csv"
OUTPUT = STUDY / "figures/dag-scaling-by-backend.png"

BATCH = 1_000_000
BUDGETS = (16, 24, 32, 48, 64, 96, 128)


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


cpu = read_csv(CPU_INPUT)
gpu = read_csv(GPU_INPUT)

cpu_lookup = {
    (int(r["components"]), int(r["budget"]), r["policy"]): r
    for r in cpu
    if int(r["batch"]) == BATCH
}

gpu_lookup = {
    (int(r["components"]), int(r["budget"])): r
    for r in gpu
    if int(r["batch"]) == BATCH
}

# Verify that every backend is evaluating the same realized DAG.
for c in range(1, 6):
    for b in BUDGETS:
        g = gpu_lookup[(c, b)]
        gpu_nodes = int(g["dag_nodes"])

        for policy in ("avx2", "avx2-omp4"):
            r = cpu_lookup[(c, b, policy)]
            cpu_nodes = int(r["nodes"])
            if cpu_nodes != gpu_nodes:
                raise RuntimeError(
                    f"DAG mismatch C{c} B{b} {policy}: "
                    f"CPU={cpu_nodes}, CUDA={gpu_nodes}"
                )

fig, axes = plt.subplots(
    1,
    5,
    figsize=(17.0, 4.8),
    sharey=True,
)

# Reserve the right margin exclusively for the legend.
fig.subplots_adjust(
    left=0.07,
    right=0.82,
    bottom=0.16,
    top=0.86,
    wspace=0.18,
)

for c, ax in enumerate(axes, start=1):
    nodes = []
    avx2 = []
    omp4 = []
    cuda = []

    for b in BUDGETS:
        a = cpu_lookup[(c, b, "avx2")]
        o = cpu_lookup[(c, b, "avx2-omp4")]
        g = gpu_lookup[(c, b)]

        nodes.append(int(g["dag_nodes"]))
        avx2.append(float(a["ns_per_eval"]))
        omp4.append(float(o["ns_per_eval"]))
        cuda.append(float(g["ns_per_eval"]))

    ax.plot(nodes, avx2, marker="o", label="AVX2")
    ax.plot(nodes, omp4, marker="s", label="AVX2 + OpenMP 4")
    ax.plot(nodes, cuda, marker="^", label="Fused CUDA")

    for x, y, b in zip(nodes, cuda, BUDGETS):
        ax.annotate(
            f"B{b}",
            (x, y),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_title(f"C{c}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)

fig.supxlabel("Realized CPWA DAG nodes")
fig.supylabel("Execution time (ns/eval, log scale)")
fig.suptitle(
    "CPWA execution scaling with DAG complexity at batch 1,000,000",
    y=0.97,
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.84, 0.52),
    ncol=1,
    frameon=False,
    title="Execution backend",
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180)
plt.close(fig)

print(OUTPUT)
