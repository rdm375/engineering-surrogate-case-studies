#!/usr/bin/env python3
"""Calibrate CPWA-ReLU GPU execution policies with resumable two-phase timing.

Each (budget, batch, dtype) point is independent and resumable.  All legal
policies receive a cheap screening pass; only the winner and near-winners are
confirmed with the requested repeat count.  Existing complete canonical
results are reused unless --force is supplied.  If every policy fails with a
resource-exhaustion error, the point is recorded and larger batches for that
budget are skipped as beyond the observed feasibility frontier.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "case-studies/thermogpu/raw-results/surrogate-performance"
BUDGETS = (16, 32, 64, 128, 256)
BATCHES = (1, 10, 100, 1000, 10000, 100000, 1000000)
RESOURCE_MARKERS = (
    "out of memory", "resource exhausted", "resource_exhausted", "oom",
    "cuda_error_out_of_memory", "failed to allocate", "allocation failed",
)


def median_ns(samples: list[float], batch: int) -> float:
    return statistics.median(samples) * 1e9 / batch


def is_resource_failure(message: str) -> bool:
    text = message.lower()
    return any(marker in text for marker in RESOURCE_MARKERS)


def reusable(path: Path, *, repeats: int, warmup: int, backend: str,
             dtype: str, budget: int, batch: int) -> tuple[bool, str | None]:
    trials_path = path.with_suffix(".trials.json")
    if not path.exists() or not trials_path.exists():
        return False, None
    try:
        profile = json.loads(path.read_text())
        detail = json.loads(trials_path.read_text())
        meta = profile.get("metadata", {})
        ok = (
            profile.get("dtype") == dtype
            and int(profile.get("batch_size", -1)) == batch
            and profile.get("backend") == backend
            and int(meta.get("budget", -1)) == budget
            and bool(meta.get("canonical_gpu_run"))
            and int(meta.get("repeats", 0)) >= repeats
            and int(meta.get("warmup", 0)) >= warmup
            and bool(detail.get("selected_policy"))
        )
        return ok, detail.get("selected_policy") if ok else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    p.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--repeats", type=int, default=7,
                   help="total timing samples for confirmed finalists")
    p.add_argument("--screen-repeats", type=int, default=2,
                   help="cheap timing samples used to screen every legal policy")
    p.add_argument("--confirm-top", type=int, default=2,
                   help="confirm at least this many fastest screening policies")
    p.add_argument("--near-tie-percent", type=float, default=10.0,
                   help="also confirm policies within this percent of screen winner")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--force", action="store_true",
                   help="ignore reusable completed calibration points")
    p.add_argument("--cpwa-relu-root", type=Path,
                   default=Path("/nvme/Sync/cpwa-relu"))
    a = p.parse_args()
    if a.repeats < 1 or a.screen_repeats < 1:
        p.error("--repeats and --screen-repeats must be positive")
    if a.screen_repeats > a.repeats:
        p.error("--screen-repeats cannot exceed --repeats")
    if a.confirm_top < 1:
        p.error("--confirm-top must be positive")
    if a.near_tie_percent < 0:
        p.error("--near-tie-percent cannot be negative")

    sys.path.insert(0, str(a.cpwa_relu_root / "src"))
    import jax
    import jax.numpy as jnp
    from cpwa_relu import load_native_dag
    from cpwa_relu.calibration import DAGHardwareProfile
    from cpwa_relu.execution_policy import default_dag_execution_policies

    if a.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    backend = jax.default_backend()
    devices = jax.devices()
    if not devices:
        raise SystemExit("JAX reports no available devices")
    device = devices[0]

    print("===== JAX EXECUTION ENVIRONMENT =====", flush=True)
    print(f"JAX version: {jax.__version__}", flush=True)
    print(f"backend:     {backend}", flush=True)
    print(f"device:      {device}", flush=True)
    print(f"devices:     {devices}", flush=True)
    print(f"dtype:       {a.dtype}", flush=True)
    print(f"screen:      {a.screen_repeats} repeats", flush=True)
    print(f"confirm:     {a.repeats} repeats; top={a.confirm_top}; near-tie={a.near_tie_percent:g}%", flush=True)
    print(flush=True)
    if backend != "gpu":
        raise SystemExit(f"refusing canonical surrogate calibration on non-GPU JAX backend: {backend}")

    rng = np.random.default_rng(a.seed)
    jd = getattr(jnp, a.dtype)
    policies = list(default_dag_execution_policies())

    for b in a.budgets:
        artifact = OUT / f"budget-{b}" / "artifacts" / f"methane-z-b{b}-native.cpwa"
        if not artifact.exists():
            raise SystemExit(f"missing native DAG: {artifact}")
        dag = load_native_dag(artifact)
        caldir = OUT / f"budget-{b}" / "calibration"
        caldir.mkdir(parents=True, exist_ok=True)
        print(f"===== BUDGET {b} =====", flush=True)
        frontier_batch: int | None = None

        for batch in sorted(a.batches):
            path = caldir / f"{a.dtype}-b{batch}.json"
            if frontier_batch is not None and batch > frontier_batch:
                print(f"SKIP budget={b:3d} batch={batch:7d}: beyond resource frontier at batch={frontier_batch}", flush=True)
                continue
            if not a.force:
                ok, selected = reusable(path, repeats=a.repeats, warmup=a.warmup,
                                        backend=backend, dtype=a.dtype, budget=b, batch=batch)
                if ok:
                    print(f"REUSE budget={b:3d} batch={batch:7d}: selected={selected}", flush=True)
                    continue

            print(f"--- batch={batch} ---", flush=True)
            x = jax.device_put(jnp.asarray(rng.random((batch, 2)), dtype=jd), device)
            x.block_until_ready()
            failures: dict[str, str] = {}
            trials: dict[str, list[float]] = {}
            phase_timings: dict[str, dict[str, float | int | str]] = {}
            executables = {}

            # Phase 1: compile every legal policy and cheaply screen steady-state cost.
            for policy in policies:
                name = policy.name
                try:
                    print(f"budget={b:3d} batch={batch:7d} policy={name}: lower/compile/screen", flush=True)
                    t0 = time.perf_counter()
                    exe = policy.make_executable(dag, dtype=a.dtype, device=device)
                    lowering_seconds = time.perf_counter() - t0
                    fn = lambda z, e=exe: e.function(e.params, z)
                    t0 = time.perf_counter(); y = fn(x); y.block_until_ready()
                    compile_seconds = time.perf_counter() - t0
                    for _ in range(a.warmup):
                        y = fn(x); y.block_until_ready()
                    ts = []
                    for _ in range(a.screen_repeats):
                        t0 = time.perf_counter(); y = fn(x); y.block_until_ready(); ts.append(time.perf_counter() - t0)
                    ns = median_ns(ts, batch)
                    trials[name] = ts
                    executables[name] = fn
                    phase_timings[name] = {
                        "lowering_seconds": lowering_seconds,
                        "compile_first_execution_seconds": compile_seconds,
                        "screen_repeats": a.screen_repeats,
                        "screen_ns_per_eval": ns,
                    }
                    print(f"budget={b:3d} batch={batch:7d} policy={name}: screen={ns:.3f} ns/eval", flush=True)
                except Exception as exc:
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    print(f"budget={b:3d} batch={batch:7d} policy={name}: FAILED: {failures[name]}", flush=True)

            if not trials:
                failure_path = caldir / f"{a.dtype}-b{batch}.failed.json"
                all_resource = bool(failures) and all(is_resource_failure(v) for v in failures.values())
                failure_path.write_text(json.dumps({
                    "status": "failed", "budget": b, "batch_size": batch,
                    "dtype": a.dtype, "backend": backend, "device": str(device),
                    "all_failures_resource_related": all_resource,
                    "policy_failures": failures,
                }, indent=2) + "\n")
                if all_resource:
                    frontier_batch = batch
                    print(f"RESOURCE FRONTIER budget={b}: all policies failed at batch={batch}; larger batches will be skipped", flush=True)
                    continue
                raise SystemExit(f"no policy succeeded for budget={b} batch={batch}")

            ranked = sorted(trials, key=lambda n: median_ns(trials[n], batch))
            best_screen = median_ns(trials[ranked[0]], batch)
            cutoff = best_screen * (1.0 + a.near_tie_percent / 100.0)
            finalists = set(ranked[: min(a.confirm_top, len(ranked))])
            finalists.update(n for n in ranked if median_ns(trials[n], batch) <= cutoff)
            print("FINALISTS " + ", ".join(n for n in ranked if n in finalists), flush=True)

            # Phase 2: add samples only for finalists; no recompilation.
            for name in ranked:
                if name not in finalists:
                    phase_timings[name]["status"] = "screened-out"
                    phase_timings[name]["ns_per_eval"] = median_ns(trials[name], batch)
                    continue
                fn = executables[name]
                needed = a.repeats - len(trials[name])
                print(f"budget={b:3d} batch={batch:7d} policy={name}: confirm +{needed}", flush=True)
                for _ in range(needed):
                    t0 = time.perf_counter(); y = fn(x); y.block_until_ready(); trials[name].append(time.perf_counter() - t0)
                ns = median_ns(trials[name], batch)
                phase_timings[name].update({
                    "status": "confirmed", "confirmed_repeats": len(trials[name]),
                    "median_execution_seconds": statistics.median(trials[name]),
                    "ns_per_eval": ns,
                })
                print(f"budget={b:3d} batch={batch:7d} policy={name}: confirmed={ns:.3f} ns/eval", flush=True)

            confirmed_costs = {n: median_ns(trials[n], batch) for n in finalists}
            selected_policy = min(confirmed_costs, key=confirmed_costs.get)
            # Profile costs remain useful for policy comparison, but selection is
            # deliberately based only on fully confirmed finalists.
            costs = {n: median_ns(ts, batch) for n, ts in trials.items()}
            profile = DAGHardwareProfile(
                name=f"thermogpu-b{b}-{a.dtype}-n{batch}", dtype=a.dtype,
                batch_size=batch, policy_costs=costs, backend=backend,
                device=str(device), policy_failures=failures,
                metadata={
                    "budget": b, "repeats": a.repeats, "screen_repeats": a.screen_repeats,
                    "confirm_top": a.confirm_top, "near_tie_percent": a.near_tie_percent,
                    "warmup": a.warmup, "seed": a.seed,
                    "artifact": str(artifact.relative_to(ROOT)), "jax_version": jax.__version__,
                    "jax_backend": backend, "jax_device": str(device),
                    "jax_devices": [str(d) for d in devices], "canonical_gpu_run": True,
                    "calibration_strategy": "screen-then-confirm-v1",
                },
            )
            profile.save(path)
            detail = {
                "profile": profile.to_dict(), "selected_policy": selected_policy,
                "selected_ns_per_eval": confirmed_costs[selected_policy],
                "screen_ranking": ranked, "confirmed_finalists": [n for n in ranked if n in finalists],
                "phase_timings": phase_timings, "trials_seconds": trials,
            }
            path.with_suffix(".trials.json").write_text(json.dumps(detail, indent=2) + "\n")
            print(f"SELECTED budget={b:3d} batch={batch:7d} {selected_policy:28s} {confirmed_costs[selected_policy]:12.3f} ns/eval", flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()
