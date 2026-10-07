#!/usr/bin/env python3
"""Engineering synthesis for fixed-composition multicomponent EOS Case Study 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
PROCESSED = STUDY / "processed-results"

BATCHES = (1, 10, 100, 1000, 10000, 100000, 1000000)
COMPONENTS = (1, 2, 3, 4, 5)


def read(name):
    with (PROCESSED / name).open(newline="") as f:
        return list(csv.DictReader(f))


def one(rows, **conditions):
    found = [
        r for r in rows
        if all(str(r[k]) == str(v) for k, v in conditions.items())
    ]
    if len(found) != 1:
        raise ValueError(
            f"expected one row for {conditions}, found {len(found)}"
        )
    return found[0]



def write_synthesis(summary: dict) -> None:
    first = summary["first_measured_surrogate_crossover"]
    through = summary["throughput_regime_batch_1000000"]

    lines = [
        "# Engineering Synthesis",
        "",
        "## Central result",
        "",
        "When the computational complexity of the underlying EOS increases "
        "while surrogate input dimension remains fixed at `(T, P)`, the "
        "economic value of the hardware-executable surrogate increases "
        "substantially.",
        "",
        "The result is workload-dependent. At batch 1, the best exact "
        "implementation dominates every measured surrogate for C1-C5. "
        "By batch 100, at least one surrogate is faster than the best exact "
        "implementation for every mixture.",
        "",
        "## First measured crossover",
        "",
        "| Mixture | Batch | Budget | RMSE | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]

    for c in range(1, 6):
        r = first[f"C{c}"]
        lines.append(
            f"| C{c} | {r['batch']:,} | {r['budget']} | "
            f"{r['rmse']:.6g} | {r['speedup']:.2f}x |"
        )

    lines += [
        "",
        "C4 and C5 already admit a narrowly faster surrogate at batch 10, "
        "although only at relatively loose measured accuracy. C1-C3 first "
        "cross over at batch 100.",
        "",
        "## Throughput regime",
        "",
        "At batch 1,000,000:",
        "",
        "| Mixture | Best exact ns/eval | Most accurate faster RMSE | "
        "Speedup at that accuracy | Maximum measured speedup |",
        "|---|---:|---:|---:|---:|",
    ]

    for c in range(1, 6):
        r = through[f"C{c}"]
        lines.append(
            f"| C{c} | {r['best_exact_ns_per_eval']:.3f} | "
            f"{r['most_accurate_faster_rmse']:.6g} | "
            f"{r['most_accurate_faster_speedup']:.2f}x | "
            f"{r['max_speedup']:.2f}x |"
        )

    lines += [
        "",
        "The maximum measured speedup rises from 45.35x for C1 to "
        "136.06x for C5. More importantly, the higher-component mixtures "
        "retain substantial speedups at much tighter accuracy than the "
        "fastest low-budget surrogate.",
        "",
        "## Why mixture complexity changes the economics",
        "",
        "The direct Peng-Robinson calculation becomes more expensive as "
        "additional mixture components are introduced. The surrogate, "
        "however, continues to approximate a two-dimensional mapping from "
        "`(T, P)` to `Z`. Its evaluation cost is therefore governed primarily "
        "by the complexity of the synthesized CPWA function rather than by "
        "the component count of the original EOS.",
        "",
        "The measured results consequently separate physical-model complexity "
        "from surrogate input dimension: increasing the former makes direct "
        "physics more expensive without causing a corresponding growth in "
        "the dimensionality of the surrogate problem.",
        "",
        "## High-accuracy hardware resource transition",
        "",
        "B96 to B128 produces a consistent hardware transition across all five "
        "mixtures. Every B96 kernel is already using 255 registers. B128 "
        "retains that register count but sharply increases local-memory "
        "spilling:",
        "",
        "| Mixture | Spill bytes B96 | Spill bytes B128 | "
        "Stack B96 | Stack B128 |",
        "|---|---:|---:|---:|---:|",
    ]

    for c in range(1, 6):
        r = summary["resource_transition_b96_to_b128"][f"C{c}"]
        lines.append(
            f"| C{c} | {r['spill_store_bytes_b96']} | "
            f"{r['spill_store_bytes_b128']} | "
            f"{r['stack_frame_bytes_b96']} | "
            f"{r['stack_frame_bytes_b128']} |"
        )

    lines += [
        "",
        "Thus additional approximation accuracy is not free even when the "
        "input dimension remains fixed. At sufficiently high CPWA complexity, "
        "the generated kernel encounters a hardware resource boundary. The "
        "observed spilling is strongly associated with the B128 execution-cost "
        "increase, although this study does not establish spilling as the sole "
        "causal mechanism.",
        "",
        "## Engineering interpretation",
        "",
        "There are therefore two distinct crossovers. The first is between "
        "direct physics and approximation: enough repeated work is required "
        "for GPU execution overhead to cease dominating surrogate cost. "
        "The second occurs within the surrogate family itself: increasing "
        "approximation complexity eventually encounters GPU resource pressure, "
        "making the next increment of accuracy disproportionately expensive.",
        "",
        "The appropriate surrogate is consequently not simply the most accurate "
        "one available. It is an operating-point decision involving required "
        "accuracy, workload size, physical-model complexity, and generated-kernel "
        "resource cost.",
        "",
        "## Scope and limitations",
        "",
    ]

    for item in summary["limitations"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "These conclusions apply to the measured fixed-composition, "
        "two-dimensional `(T, P)` problem and should not be generalized "
        "directly to variable-composition EOS surrogates.",
        "",
    ]

    out = STUDY / "ENGINEERING-SYNTHESIS.md"
    out.write_text("\n".join(lines))
    print(out)

def main():
    exact = read("exact-crossover.csv")
    surrogate = read("surrogate-performance.csv")
    crossover = read("surrogate-crossover.csv")
    resources = read("resource-pressure.csv")
    pareto = read("pareto-frontiers.csv")

    assert len(exact) == 35
    assert len(surrogate) == 245
    assert len(crossover) == 35
    assert len(resources) == 35

    # First measured batch at which any surrogate beats exact.
    first_crossover = {}
    for c in COMPONENTS:
        rows = sorted(
            (
                r for r in crossover
                if int(r["components"]) == c
            ),
            key=lambda r: int(r["batch"]),
        )
        winner = next(
            (r for r in rows if r["any_surrogate_faster"] == "True"),
            None,
        )
        assert winner is not None
        first_crossover[f"C{c}"] = {
            "batch": int(winner["batch"]),
            "budget": int(winner["most_accurate_faster_budget"]),
            "rmse": float(winner["most_accurate_faster_rmse"]),
            "speedup": float(winner["most_accurate_faster_speedup"]),
        }

    # Throughput-regime summary at batch 1,000,000.
    throughput = {}
    for c in COMPONENTS:
        r = one(crossover, components=c, batch=1000000)
        throughput[f"C{c}"] = {
            "best_exact_ns_per_eval":
                float(r["best_exact_ns_per_eval"]),
            "most_accurate_faster_budget":
                int(r["most_accurate_faster_budget"]),
            "most_accurate_faster_rmse":
                float(r["most_accurate_faster_rmse"]),
            "most_accurate_faster_speedup":
                float(r["most_accurate_faster_speedup"]),
            "fastest_surrogate_budget":
                int(r["fastest_surrogate_budget"]),
            "fastest_surrogate_rmse":
                float(r["fastest_surrogate_rmse"]),
            "fastest_surrogate_ns_per_eval":
                float(r["fastest_surrogate_ns_per_eval"]),
            "max_speedup":
                float(r["max_speedup"]),
        }

    # Exact physical-model cost scaling at the throughput asymptote.
    exact_ns = {
        f"C{c}": float(
            one(exact, components=c, batch=1000000)["ns_per_eval"]
        )
        for c in COMPONENTS
    }

    # Surrogate cost scaling for representative budgets at batch 1e6.
    surrogate_ns = {}
    for budget in (16, 64, 96, 128):
        surrogate_ns[f"B{budget}"] = {
            f"C{c}": float(
                one(
                    surrogate,
                    components=c,
                    budget=budget,
                    batch=1000000,
                )["ns_per_eval"]
            )
            for c in COMPONENTS
        }

    # Approximation geometry at the largest budget.
    geometry_b128 = {}
    for c in COMPONENTS:
        r = one(resources, components=c, budget=128)
        geometry_b128[f"C{c}"] = {
            "rmse": float(r["rmse"]),
            "points": int(r["points"]),
            "simplices": int(r["simplices"]),
            "dag_nodes": int(r["dag_nodes"]),
        }

    # Resource transition B96 -> B128.
    resource_transition = {}
    for c in COMPONENTS:
        b96 = one(resources, components=c, budget=96)
        b128 = one(resources, components=c, budget=128)

        resource_transition[f"C{c}"] = {
            "rmse_b96": float(b96["rmse"]),
            "rmse_b128": float(b128["rmse"]),
            "registers_b96": int(b96["registers"]),
            "registers_b128": int(b128["registers"]),
            "spill_store_bytes_b96": int(b96["spill_store_bytes"]),
            "spill_store_bytes_b128": int(b128["spill_store_bytes"]),
            "stack_frame_bytes_b96": int(b96["stack_frame_bytes"]),
            "stack_frame_bytes_b128": int(b128["stack_frame_bytes"]),
        }

    # Number of global Pareto points in each operating condition.
    pareto_counts = {}
    for c in COMPONENTS:
        pareto_counts[f"C{c}"] = {
            str(batch): sum(
                1
                for r in pareto
                if int(r["components"]) == c
                and int(r["batch"]) == batch
            )
            for batch in BATCHES
        }

    summary = {
        "study": "Fixed-Composition Multicomponent EOS",
        "case_study": 2,
        "central_question": (
            "How does the economic value of a hardware-executable "
            "surrogate change as computational complexity of the "
            "underlying physical model increases while surrogate input "
            "dimension remains fixed?"
        ),
        "components": list(COMPONENTS),
        "batches": list(BATCHES),
        "surrogate_input_dimension": 2,
        "surrogate_inputs": ["temperature", "pressure"],
        "first_measured_surrogate_crossover": first_crossover,
        "throughput_regime_batch_1000000": throughput,
        "best_exact_ns_per_eval_batch_1000000": exact_ns,
        "surrogate_ns_per_eval_batch_1000000": surrogate_ns,
        "geometry_at_budget_128": geometry_b128,
        "resource_transition_b96_to_b128": resource_transition,
        "pareto_point_counts": pareto_counts,
        "pareto_total_points": len(pareto),
        "interpretation": {
            "scalar_latency": (
                "At batch 1, the best exact implementation dominates "
                "all measured surrogates for C1-C5."
            ),
            "workload_crossover": (
                "At batch 100, at least one measured surrogate is "
                "faster than the best exact implementation for every "
                "mixture C1-C5."
            ),
            "complexity_effect": (
                "As component count increases while surrogate input "
                "dimension remains fixed at (T,P), direct EOS cost "
                "increases substantially while surrogate evaluation "
                "cost remains comparatively insensitive to component "
                "count."
            ),
            "resource_limit": (
                "At B96 all five measured surrogate kernels use 255 "
                "registers. Moving to B128 sharply increases spilling "
                "for every mixture, coincident with a substantial "
                "high-accuracy execution-cost penalty."
            ),
        },
        "limitations": [
            "Composition is fixed for each mixture; the surrogate input space is only (T,P).",
            "The distinct C1-C5 family changes both component count and species identity.",
            "Binary interaction coefficients are zero.",
            "The surrogate predicts Z, while the exact EOS path computes richer thermodynamic output.",
            "Accuracy thresholds are selected only from measured TDAR budgets; no interpolation is used.",
            "Direct and surrogate resident-GPU timing harnesses have comparable intent but are not identical.",
            "Results represent one measured hardware/software environment.",
            "Register spilling is strongly associated with the B128 slowdown but is not established here as the sole causal mechanism.",
        ],
    }

    out = PROCESSED / "study-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out)

    write_synthesis(summary)

    print("study synthesis: PASS")


if __name__ == "__main__":
    main()
