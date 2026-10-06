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
