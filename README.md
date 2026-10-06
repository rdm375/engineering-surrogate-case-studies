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

See `case-studies/thermogpu/SPEC.md`.

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
