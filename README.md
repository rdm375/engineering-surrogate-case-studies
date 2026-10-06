# Engineering Surrogate Case Studies

Reproducible engineering case studies comparing direct numerical models with
hardware-accelerated surrogate representations.

The central question is:

> When should an engineering calculation be evaluated directly on CPU/GPU,
> and when is it advantageous to replace repeated evaluations with an
> adaptively constructed hardware-accelerated surrogate?

This repository integrates separately developed and versioned numerical tools.
It is an experimental and evidence repository, not a development repository
for those tools.

## Case studies

### 1. Thermodynamic property evaluation

The first case study compares repeated Peng--Robinson methane compressibility
factor calculations using:

- scalar CPU direct evaluation;
- parallel CPU direct evaluation;
- CUDA GPU direct evaluation; and
- TDAR -> simplicial CPWA -> CPWA-ReLU surrogate evaluation.

The study measures numerical error, latency, throughput, batch scaling,
surrogate construction cost, and amortization.

Frozen software stack:

- ThermoGPU v1.1.0
- TDAR v0.3.0
- CPWA-ReLU v0.2.0

V1 result: on the measured methane-Z problem and hardware, the surrogate did not
outperform the best workload-conditioned exact implementation. All 31 measured
surrogate operating points were dominated by the fastest exact method at the
same batch size, and no surrogate point was globally Pareto-optimal. This is a
measured negative result for this case, not a general claim about surrogates.

See `case-studies/thermogpu/reports/thermogpu-v1.md` for the generated V1
report and `case-studies/thermogpu/SPEC.md` for the frozen study specification.

### 2. Compressor station

Planned application of the established methodology to a multivariable
engineering equipment model.

Not yet started.

### 3. Transient pipe

Planned application of the established methodology to dynamic pipe-flow
simulation, where repeated constitutive and thermodynamic evaluations make
surrogate amortization particularly important.

Not yet started.

## Repository principle

Released upstream components are treated as frozen dependencies during a case
study.

This repository measures the stack; it does not develop the stack.

Correctness defects discovered during a case study should be fixed upstream
and incorporated through a new released dependency version. Performance
results that motivate new algorithms are recorded as results rather than
silently expanding the scope of the current case study.

A missing Pareto point does not create a new research project.

## Study framework

Case-study repositories own orchestration, evidence normalization, provenance,
Pareto analysis, plotting, and reporting. Released upstream projects remain the
authoritative implementations of the numerical methods themselves.

The ThermoGPU study is configured by
`case-studies/thermogpu/configs/study-v1.toml`. Shared result and provenance
infrastructure lives in `src/engineering_case_studies/`; ThermoGPU-specific
collection scripts live under `case-studies/thermogpu/scripts/`.

The intended boundary is deliberately thin: collectors invoke frozen upstream
releases and retain their raw evidence; processing and reporting operate on the
retained case-study data rather than reimplementing ThermoGPU, TDAR, or
CPWA-ReLU.

## Local and Slurm execution

Case-study jobs support two execution environments: direct local execution for
workstations and Slurm submission for HPC systems. Slurm is treated as an
experiment-execution backend, not as a dependency of ThermoGPU, TDAR, or
CPWA-ReLU. The same retained raw-result and reporting formats are used in both
cases.

ThermoGPU Slurm templates live in `case-studies/thermogpu/slurm/`. They use job
arrays for batch-size and surrogate-budget sweeps and deliberately avoid
hard-coding site-specific partitions, accounts, or QOS settings. Those may be
provided at submission time. `submit_study.py` records the returned Slurm job
ID and submission command under `raw-results/slurm/` for provenance.

The Slurm templates reference the case-study collectors; as those collectors
are implemented, local and cluster execution will therefore exercise the same
measurement code rather than parallel benchmark implementations.

## Automated ThermoGPU V1 pipeline

The complete local experiment is orchestrated by
`case-studies/thermogpu/scripts/run_study.py`. It reads the frozen budgets and
batch sizes from `configs/study-v1.toml`, preserves the single shared TDAR
validation sweep, builds exact native DAGs, calibrates all legal CPWA-ReLU GPU
execution policies, benchmarks the empirically selected policy, processes the
retained evidence, regenerates figures, and writes the Markdown report.

The pipeline is resumable: a stage whose declared outputs are already present
is skipped. Use `--force` to rerun selected stages, `--from-stage` and
`--through-stage` for partial runs, `--list-stages` to inspect the stage graph,
and `--dry-run` to print commands without executing them. GPU calibration and
surrogate benchmarking use `--cpwa-python` (by default the CPWA-ReLU virtual
environment) so they cannot silently fall back to the case-study environment's
CPU-only Python. The canonical GPU calibration protocol screens every legal policy with 2 timing samples, then confirms at least the 2 fastest policies (plus any within 10% of the screening winner) using the configured canonical repeat count.


## Reproducing ThermoGPU V1 from retained evidence

The V1 scientific products can be regenerated without rerunning the expensive
evidence-producing experiments. From a checkout with the frozen upstream
dependencies available, run:

```bash
<cpwa-relu-python> case-studies/thermogpu/scripts/run_study.py \
  --config case-studies/thermogpu/configs/study-v1.toml \
  --reproduce
```

Reproduction mode validates and retains experimental evidence, then regenerates
processed results, publication figures, Pareto analyses, engineering analysis,
and the final Markdown report. The historical `performance-process` and
`performance-plot` stages are intentionally omitted in this mode: the
authoritative V1 analysis consumes retained per-cell calibration evidence
directly.
