#!/usr/bin/env python3
"""Calibrate every legal CPWA-ReLU native-DAG execution policy per budget/batch.

Canonical case-study calibration is GPU-only.

For each execution policy, separately measure:

1. DAG scheduling/lowering time;
2. first-call JAX/XLA compilation plus first execution time;
3. steady-state GPU execution time.

The first call is deliberately excluded from the steady-state timing samples.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(BUDGETS),
    )
    p.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=list(BATCHES),
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
    )
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument(
        "--cpwa-relu-root",
        type=Path,
        default=Path("/nvme/Sync/cpwa-relu"),
    )
    a = p.parse_args()

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
    print(flush=True)

    if backend != "gpu":
        raise SystemExit(
            "refusing canonical surrogate calibration on "
            f"non-GPU JAX backend: {backend}"
        )

    rng = np.random.default_rng(a.seed)
    jd = getattr(jnp, a.dtype)

    for b in a.budgets:
        artifact = (
            OUT
            / f"budget-{b}"
            / "artifacts"
            / f"methane-z-b{b}-native.cpwa"
        )

        if not artifact.exists():
            raise SystemExit(f"missing native DAG: {artifact}")

        dag = load_native_dag(artifact)

        caldir = OUT / f"budget-{b}" / "calibration"
        caldir.mkdir(parents=True, exist_ok=True)

        print(
            f"===== BUDGET {b} =====",
            flush=True,
        )

        for batch in a.batches:
            print(
                f"--- batch={batch} ---",
                flush=True,
            )

            x = jax.device_put(
                jnp.asarray(
                    rng.random((batch, 2)),
                    dtype=jd,
                ),
                device,
            )

            # Ensure input placement itself is complete before calibration.
            x.block_until_ready()

            costs = {}
            failures = {}
            trials = {}
            phase_timings = {}

            for policy in default_dag_execution_policies():
                name = policy.name

                try:
                    print(
                        f"budget={b:3d} batch={batch:7d} "
                        f"policy={name}: scheduling/lowering",
                        flush=True,
                    )

                    t0 = time.perf_counter()
                    exe = policy.make_executable(
                        dag,
                        dtype=a.dtype,
                        device=device,
                    )
                    lowering_seconds = time.perf_counter() - t0

                    fn = lambda z, e=exe: e.function(e.params, z)

                    print(
                        f"budget={b:3d} batch={batch:7d} "
                        f"policy={name}: JIT compile/first execution",
                        flush=True,
                    )

                    t0 = time.perf_counter()
                    y = fn(x)
                    y.block_until_ready()
                    compile_seconds = time.perf_counter() - t0

                    print(
                        f"budget={b:3d} batch={batch:7d} "
                        f"policy={name}: warmup x{a.warmup}",
                        flush=True,
                    )

                    for _ in range(a.warmup):
                        y = fn(x)
                        y.block_until_ready()

                    print(
                        f"budget={b:3d} batch={batch:7d} "
                        f"policy={name}: benchmark x{a.repeats}",
                        flush=True,
                    )

                    ts = []

                    for _ in range(a.repeats):
                        t0 = time.perf_counter()
                        y = fn(x)
                        y.block_until_ready()
                        ts.append(time.perf_counter() - t0)

                    ns = statistics.median(ts) * 1e9 / batch

                    costs[name] = ns
                    trials[name] = ts
                    phase_timings[name] = {
                        "lowering_seconds": lowering_seconds,
                        "compile_first_execution_seconds": compile_seconds,
                        "median_execution_seconds": statistics.median(ts),
                        "ns_per_eval": ns,
                    }

                    print(
                        f"budget={b:3d} batch={batch:7d} "
                        f"policy={name}: "
                        f"lower={lowering_seconds:.3f}s "
                        f"compile={compile_seconds:.3f}s "
                        f"steady={ns:.3f} ns/eval",
                        flush=True,
                    )

                except Exception as exc:
                    failures[name] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    print(
                        f"budget={b:3d} batch={batch:7d} "
                        f"policy={name}: FAILED: {failures[name]}",
                        flush=True,
                    )

            if not costs:
                raise SystemExit(
                    f"no policy succeeded for budget={b} batch={batch}"
                )

            selected_policy = min(costs, key=costs.get)

            profile = DAGHardwareProfile(
                name=f"thermogpu-b{b}-{a.dtype}-n{batch}",
                dtype=a.dtype,
                batch_size=batch,
                policy_costs=costs,
                backend=backend,
                device=str(device),
                policy_failures=failures,
                metadata={
                    "budget": b,
                    "repeats": a.repeats,
                    "warmup": a.warmup,
                    "seed": a.seed,
                    "artifact": str(artifact.relative_to(ROOT)),
                    "jax_version": jax.__version__,
                    "jax_backend": backend,
                    "jax_device": str(device),
                    "jax_devices": [str(d) for d in devices],
                    "canonical_gpu_run": True,
                },
            )

            path = caldir / f"{a.dtype}-b{batch}.json"
            profile.save(path)

            detail = {
                "profile": profile.to_dict(),
                "selected_policy": selected_policy,
                "selected_ns_per_eval": costs[selected_policy],
                "phase_timings": phase_timings,
                "trials_seconds": trials,
            }

            path.with_suffix(".trials.json").write_text(
                json.dumps(detail, indent=2) + "\n"
            )

            print(
                f"SELECTED budget={b:3d} batch={batch:7d} "
                f"{selected_policy:28s} "
                f"{costs[selected_policy]:12.3f} ns/eval",
                flush=True,
            )
            print(flush=True)


if __name__ == "__main__":
    main()
