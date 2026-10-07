#!/usr/bin/env python3
"""Case Study 2 orchestration.

--reproduce performs evidence-only deterministic reproduction:
  * verifies retained raw evidence
  * never executes ThermoGPU, TDAR, CPWA synthesis, or CUDA benchmarks
  * removes only derived artifacts
  * rebuilds processed results, figures, and engineering synthesis
  * validates the regenerated study
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "case-studies/thermogpu-fixed-mixture"
RAW = STUDY / "raw-results"
PROCESSED = STUDY / "processed-results"
FIGURES = STUDY / "figures"
SCRIPTS = STUDY / "scripts"

DERIVED_FILES = (
    PROCESSED / "exact-performance.csv",
    PROCESSED / "exact-crossover.csv",
    PROCESSED / "surrogate-performance.csv",
    PROCESSED / "combined-candidates.csv",
    PROCESSED / "pareto-frontiers.csv",
    PROCESSED / "surrogate-crossover.csv",
    PROCESSED / "resource-pressure.csv",
    PROCESSED / "study-summary.json",
    STUDY / "ENGINEERING-SYNTHESIS.md",
    STUDY / "reports/thermogpu-fixed-mixture-v1.md",
)

FIGURE_FILES = (
    FIGURES / "accuracy-vs-budget.png",
    FIGURES / "component-scaling.png",
    FIGURES / "pareto-by-mixture.png",
    FIGURES / "resource-pressure.png",
    FIGURES / "speedup-vs-components.png",
    FIGURES / "surrogate-crossover-threshold.png",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_evidence() -> None:
    manifest = RAW / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError(f"missing evidence manifest: {manifest}")

    entries = []

    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = STUDY / relative

        if path.resolve() == manifest.resolve():
            raise RuntimeError("SHA256SUMS must not contain itself")

        if not path.is_file():
            raise RuntimeError(f"missing retained evidence: {relative}")

        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(
                f"evidence checksum mismatch: {relative}\n"
                f"expected {digest}\n"
                f"actual   {actual}"
            )

        entries.append(relative)

    if len(entries) != 8:
        raise RuntimeError(
            f"expected 8 retained evidence files, found {len(entries)}"
        )

    print(f"evidence verification: PASS ({len(entries)} files)")


def run(script: str) -> None:
    path = SCRIPTS / script
    print(f"\n===== {script} =====", flush=True)
    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )


def clean_derived() -> None:
    for path in (*DERIVED_FILES, *FIGURE_FILES):
        if path.exists():
            path.unlink()
            print(f"removed derived artifact: {path.relative_to(ROOT)}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validate() -> None:
    missing = [
        p.relative_to(ROOT)
        for p in (*DERIVED_FILES, *FIGURE_FILES)
        if not p.is_file()
    ]
    if missing:
        raise RuntimeError(f"missing derived artifacts: {missing}")

    exact = read_csv(PROCESSED / "exact-performance.csv")
    exact_cross = read_csv(PROCESSED / "exact-crossover.csv")
    surrogate = read_csv(PROCESSED / "surrogate-performance.csv")
    combined = read_csv(PROCESSED / "combined-candidates.csv")
    pareto = read_csv(PROCESSED / "pareto-frontiers.csv")
    crossover = read_csv(PROCESSED / "surrogate-crossover.csv")
    resources = read_csv(PROCESSED / "resource-pressure.csv")

    expected = {
        "exact-performance": (len(exact), 245),
        "exact-crossover": (len(exact_cross), 35),
        "surrogate-performance": (len(surrogate), 245),
        "combined-candidates": (len(combined), 490),
        "pareto-frontiers": (len(pareto), 206),
        "surrogate-crossover": (len(crossover), 35),
        "resource-pressure": (len(resources), 35),
    }

    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise RuntimeError(
                f"{name}: expected {wanted} rows, found {actual}"
            )

    groups = {}
    for r in combined:
        key = (int(r["components"]), int(r["batch"]))
        groups.setdefault(key, []).append(r)

    if len(groups) != 35:
        raise RuntimeError(
            f"expected 35 Pareto operating conditions, found {len(groups)}"
        )

    for key, rows in groups.items():
        if len(rows) != 14:
            raise RuntimeError(
                f"{key}: expected 14 candidates, found {len(rows)}"
            )

        exact_rows = [
            r for r in rows if r["candidate_type"] == "exact"
        ]
        surrogate_rows = [
            r for r in rows if r["candidate_type"] == "surrogate"
        ]

        if len(exact_rows) != 7 or len(surrogate_rows) != 7:
            raise RuntimeError(
                f"{key}: expected 7 exact + 7 surrogate candidates"
            )

        if sum(r["pareto"] == "True" for r in exact_rows) != 1:
            raise RuntimeError(
                f"{key}: expected exactly one Pareto-optimal exact method"
            )

    summary = json.loads(
        (PROCESSED / "study-summary.json").read_text()
    )

    if summary["case_study"] != 2:
        raise RuntimeError("study-summary case-study identity mismatch")
    if summary["surrogate_input_dimension"] != 2:
        raise RuntimeError("surrogate input dimension mismatch")
    if summary["pareto_total_points"] != 206:
        raise RuntimeError("Pareto total mismatch")

    first = summary["first_measured_surrogate_crossover"]
    expected_first = {
        "C1": 100,
        "C2": 100,
        "C3": 100,
        "C4": 10,
        "C5": 10,
    }

    for c, batch in expected_first.items():
        if first[c]["batch"] != batch:
            raise RuntimeError(
                f"{c}: first crossover changed from expected batch {batch}"
            )

    synthesis = STUDY / "ENGINEERING-SYNTHESIS.md"
    if synthesis.stat().st_size == 0:
        raise RuntimeError("engineering synthesis is empty")

    report = STUDY / "reports/thermogpu-fixed-mixture-v1.md"
    if report.stat().st_size == 0:
        raise RuntimeError("final report is empty")

    for figure in FIGURE_FILES:
        if figure.stat().st_size == 0:
            raise RuntimeError(f"empty figure: {figure.name}")

    print("\n===== VALIDATION =====")
    for name, (actual, _) in expected.items():
        print(f"{name}: {actual} rows PASS")
    print("Pareto operating conditions: 35 PASS")
    print("derived figures: 6 PASS")
    print("study summary: PASS")
    print("engineering synthesis: PASS")
    print("final report: PASS")


def reproduce() -> None:
    print("Case Study 2 evidence-only reproduction")
    print(f"study: {STUDY.relative_to(ROOT)}")
    print(
        "policy: retained raw evidence is immutable; "
        "no evidence producers will be executed"
    )

    verify_evidence()
    clean_derived()

    run("process_exact.py")
    run("process_surrogates.py")
    run("process_pareto.py")
    run("process_crossover_resources.py")
    run("analyze_study.py")
    run("plot_study.py")
    run("generate_report.py")

    validate()

    print("\nREPRODUCTION: PASS")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reproduce",
        action="store_true",
        help="rebuild all derived results from retained evidence",
    )
    args = p.parse_args()

    if not args.reproduce:
        p.error(
            "no action requested; use --reproduce for evidence-only "
            "reproduction"
        )

    reproduce()


if __name__ == "__main__":
    main()
