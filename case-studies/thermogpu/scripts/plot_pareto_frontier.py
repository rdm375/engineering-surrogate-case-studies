#!/usr/bin/env python3
"""Plot ThermoGPU batch-conditioned and conventional Pareto frontiers."""

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


def conventional_pareto(rows):
    """Nondominated set in (ns/eval, L_inf), independent of batch."""
    frontier = []

    for r in rows:
        x = float(r["ns_per_eval"])
        y = float(r["linf"])

        dominated = False

        for q in rows:
            if q is r:
                continue

            qx = float(q["ns_per_eval"])
            qy = float(q["linf"])

            if (
                qx <= x
                and qy <= y
                and (qx < x or qy < y)
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(r)

    return sorted(
        frontier,
        key=lambda r: (
            float(r["ns_per_eval"]),
            float(r["linf"]),
        ),
    )


def surrogate_pareto(rows):
    """Conventional nondominated set restricted to surrogates."""
    return conventional_pareto(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()

    rows = [
        r for r in read_rows(args.input)
        if truth(r["feasible"])
    ]

    if not rows:
        raise SystemExit("master performance terrain is empty")

    exact = [r for r in rows if truth(r["exact"])]
    surrogate = [r for r in rows if not truth(r["exact"])]

    # Existing processor flag: best exact method conditioned on batch.
    batch_frontier = [
        r for r in rows
        if truth(r["global_pareto"])
    ]

    # True two-objective Pareto sets, independent of batch.
    global_frontier = conventional_pareto(rows)
    surrogate_frontier = surrogate_pareto(surrogate)

    by_method = defaultdict(list)

    for r in rows:
        by_method[r["method"]].append(r)

    for rs in by_method.values():
        rs.sort(key=lambda r: int(r["batch_size"]))

    surrogate_by_budget = defaultdict(list)

    for r in surrogate:
        surrogate_by_budget[int(r["budget"])].append(r)

    for rs in surrogate_by_budget.values():
        rs.sort(key=lambda r: int(r["batch_size"]))

    # ===============================================================
    # Figure
    # ===============================================================

    fig, (ax_all, ax_sur) = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.25, 1.0)},
    )

    # ===============================================================
    # A. Complete performance terrain
    # ===============================================================

    for method, rs in sorted(by_method.items()):
        is_surrogate = rs[0]["backend"] == "gpu_surrogate"

        ax_all.plot(
            [int(r["batch_size"]) for r in rs],
            [float(r["ns_per_eval"]) for r in rs],
            marker="o",
            linewidth=1.25 if is_surrogate else 2.0,
            markersize=4 if is_surrogate else 5.5,
            alpha=0.58 if is_surrogate else 0.92,
            label=method,
        )

    # Fastest exact implementation at each fixed batch size.
    ax_all.scatter(
        [int(r["batch_size"]) for r in batch_frontier],
        [float(r["ns_per_eval"]) for r in batch_frontier],
        marker="*",
        s=190,
        linewidths=0.9,
        zorder=10,
        label="Batch-conditioned best-exact frontier",
    )

    # Conventional global Pareto set.
    #
    # This is evaluated in (ns/eval, L_inf) space without conditioning
    # on batch. Mark its members separately on the scaling plot.
    ax_all.scatter(
        [int(r["batch_size"]) for r in global_frontier],
        [float(r["ns_per_eval"]) for r in global_frontier],
        marker="D",
        s=75,
        facecolors="none",
        linewidths=1.8,
        zorder=11,
        label="Conventional global Pareto set",
    )

    ax_all.set_xscale("log")
    ax_all.set_yscale("log")
    ax_all.set_xlabel("Batch size")
    ax_all.set_ylabel("Nanoseconds per evaluation")
    ax_all.set_title(
        "A. Performance terrain and exact implementation frontier"
    )
    ax_all.grid(True, which="both", alpha=0.22)

    ax_all.legend(
        fontsize=7.5,
        ncol=2,
        loc="best",
    )

    # ===============================================================
    # B. Conventional surrogate Pareto frontier
    # ===============================================================

    for budget, rs in sorted(surrogate_by_budget.items()):
        ax_sur.plot(
            [float(r["ns_per_eval"]) for r in rs],
            [float(r["linf"]) for r in rs],
            marker="o",
            linewidth=1.35,
            markersize=5,
            alpha=0.55,
            label=f"budget {budget}",
        )

    # Highlight only points that are genuinely nondominated when batch
    # is NOT treated as a constraint.
    ax_sur.scatter(
        [float(r["ns_per_eval"]) for r in surrogate_frontier],
        [float(r["linf"]) for r in surrogate_frontier],
        s=105,
        facecolors="none",
        linewidths=1.8,
        zorder=10,
        label="Conventional surrogate Pareto frontier",
    )

    # Connect the conventional surrogate frontier in performance order.
    if len(surrogate_frontier) > 1:
        sf = sorted(
            surrogate_frontier,
            key=lambda r: float(r["ns_per_eval"]),
        )

        ax_sur.plot(
            [float(r["ns_per_eval"]) for r in sf],
            [float(r["linf"]) for r in sf],
            linewidth=2.2,
            linestyle="--",
            alpha=0.8,
        )

    ax_sur.set_xscale("log")
    ax_sur.set_yscale("log")

    ax_sur.set_xlabel("Nanoseconds per evaluation")
    ax_sur.set_ylabel(
        r"$L_\infty$ absolute error in compressibility factor $Z$"
    )

    ax_sur.set_title(
        "B. Conventional surrogate Pareto frontier"
    )

    ax_sur.grid(True, which="both", alpha=0.22)

    ax_sur.annotate(
        "Better: down and left",
        xy=(0.72, 0.72),
        xytext=(0.92, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    ax_sur.legend(
        title="TDAR budget",
        fontsize=8,
        title_fontsize=8,
        loc="best",
    )

    fig.suptitle(
        "ThermoGPU methane Z: performance and Pareto structure",
        fontsize=15,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(args.output)

    print(
        f"records={len(rows)} "
        f"exact={len(exact)} "
        f"surrogate={len(surrogate)}"
    )

    print(
        "batch-conditioned best-exact frontier: "
        f"{len(batch_frontier)} points"
    )

    print(
        "conventional surrogate Pareto frontier: "
        f"{len(surrogate_frontier)} points"
    )

    print(
        "conventional global Pareto frontier: "
        f"{len(global_frontier)} points"
    )

    print()
    print("CONVENTIONAL GLOBAL PARETO SET")

    for r in global_frontier:
        print(
            f"  {r['method']:24s} "
            f"batch={int(r['batch_size']):7d} "
            f"ns/eval={float(r['ns_per_eval']):12.6f} "
            f"Linf={float(r['linf']):.8g}"
        )

    print()
    print("CONVENTIONAL SURROGATE PARETO SET")

    for r in surrogate_frontier:
        print(
            f"  budget={int(r['budget']):3d} "
            f"batch={int(r['batch_size']):7d} "
            f"ns/eval={float(r['ns_per_eval']):12.6f} "
            f"Linf={float(r['linf']):.8g}"
        )


if __name__ == "__main__":
    main()
