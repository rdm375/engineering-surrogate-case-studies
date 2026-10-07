# Case Study 2 Evidence Inventory

## Purpose

This document inventories the retained upstream evidence for Case Study 2,
Fixed-Composition Multicomponent EOS.

The raw evidence is copied into this repository so that deterministic
analysis and reproduction do not depend on the mutable state of the
upstream ThermoGPU working tree.

## Upstream provenance

### ThermoGPU

Evidence branch:

    case-study-2-pareto-benchmark

Base release:

    v1.1.0
    180bbcb0eb70747037013dc3ea44032e10185f74

Exact-scaling evidence:

    e233f02
    Add distinct-mixture exact scaling benchmark

Surrogate hardware evidence:

    1c95ebc
    Complete distinct-mixture surrogate hardware sweep

The ThermoGPU test suite passed 9/9 after evidence acquisition.

### CPWA-ReLU

Release:

    v0.2.0

Commit:

    6ac34968992ae17c079f92ba37c0deacd5b69a19

## Exact-performance evidence

Directory:

    raw-results/exact/

Files:

### exact-scaling.csv

SHA-256:

    f8a7cae49f536a5bcde0e8314698dc4f7fda6040982253055b55f7b0eb9fc385

Contains 245 measurements:

    5 mixtures
    × 7 batch sizes
    × 7 exact execution methods
    = 245 rows

Execution methods:

- scalar CPU
- OpenMP 1 thread
- OpenMP 2 threads
- OpenMP 4 threads
- OpenMP 8 threads
- generic resident CUDA
- fixed-composition resident CUDA

Batch sizes:

- 1
- 10
- 100
- 1,000
- 10,000
- 100,000
- 1,000,000

### provenance.txt

SHA-256:

    2cf46955f178b3e529f9b14a92ebc91766861231f41758f0f8b5bb566071d76e

Records the exact benchmark acquisition provenance.

### benchmark.log

SHA-256:

    32b0ebcb589d57d14d88df51c5b980d1987ed0eb83d1846cb286027009d91a41

Retains the exact benchmark execution log.

## Surrogate hardware evidence

Directory:

    raw-results/surrogates/

The hardware sweep contains:

    5 mixtures
    × 7 approximation budgets
    × 7 batch sizes
    = 245 measurements

There are 35 distinct frozen approximation artifacts.

Budgets:

- 16
- 24
- 32
- 48
- 64
- 96
- 128

Batch sizes:

- 1
- 10
- 100
- 1,000
- 10,000
- 100,000
- 1,000,000

### summary.csv

SHA-256:

    f13062291adc1e922039d53aefd5de9083550fce9d1444bca92d679c5177d61f

Primary tabular surrogate hardware evidence.

For each `(mixture, budget, batch)` point it retains approximation,
execution, DAG, and hardware-resource quantities including:

- RMSE
- p99 absolute error
- maximum absolute error
- TDAR point count
- simplex count
- execution time
- DAG node counts
- executed operations
- live slots
- registers
- spill stores
- spill loads
- stack-frame size
- CPU/CUDA validation delta

All 245 retained rows have:

    max_delta_cpu = 0

### summary.json

SHA-256:

    281ee272c5ad574f0cc791671848a0348cac165336e3ba6f07596ed813cdc3fd

JSON representation of the aggregate hardware sweep.

### metadata.json

SHA-256:

    d7bb73fed29ec3f374e092c6709b88970917a2dbb0d7a0c12fe37fbfa824729e

Records sweep configuration and selection metadata.

### provenance.txt

SHA-256:

    8abe320771b5a30343c8f7088c1b94f7d20f4a181ed1b188830de04144ae8bf2

Records upstream revisions and hardware/software provenance.

### run.log

SHA-256:

    ff47ae7fe5b74daf110e79c8eb2b751db2d928f0bdafd119b03a2e0b723e3ae5

Complete retained acquisition log.

## Integrity manifest

File:

    raw-results/SHA256SUMS

The manifest covers all eight imported upstream evidence files.

Verification command:

    (
      cd case-studies/thermogpu-fixed-mixture
      sha256sum -c raw-results/SHA256SUMS
    )

All eight files passed checksum verification at import.

## Structural validation

Import validation established:

    exact rows:                 245
    exact operating points:     35
    exact methods/point:          7

    surrogate rows:             245
    surrogate artifacts:         35
    surrogate batches/artifact:   7

    CPU/CUDA validation:        PASS
    evidence import:            PASS

## Evidence/derivation boundary

Files under `raw-results/` are retained evidence.

Deterministic case-study processing MUST NOT modify these files.

Derived tables belong under:

    processed-results/

Derived figures belong under:

    figures/

Derived reports belong under:

    reports/

Reproduction may verify retained evidence and regenerate derived artifacts,
but must not silently reacquire or replace the frozen evidence.
