# ThermoGPU Engineering Surrogate Case Study — V1 Milestones

## Objective

Determine where direct thermodynamic computation and surrogate evaluation are
advantageous across workload size, hardware execution strategy, surrogate
complexity, and approximation accuracy.

V1 is complete when every experiment necessary to support or falsify the
engineering conclusions has been completed. V1 does not require populating
every possible budget × batch combination when a combination is demonstrably
infeasible or cannot materially affect the conclusions.

## Milestones

| ID | Milestone | Status | V1 evidence |
|---|---|---|---|
| M1 | Define and freeze V1 experiment | Complete | `configs/study-v1.toml` |
| M2 | Validate direct ThermoGPU calculation | Complete | Direct validation evidence |
| M3 | Benchmark direct execution methods | Complete | CPU scalar, CPU OpenMP, GPU resident, GPU E2E |
| M4 | Build TDAR surrogate family | Complete | Budgets 16, 32, 64, 128, 256 |
| M5 | Validate surrogate accuracy | Complete | Shared frozen validation set; RMSE, p99, Linf |
| M6 | Compile exact CPWA-ReLU DAGs | Complete | Native `.cpwa` artifacts |
| M7 | Calibrate GPU surrogate execution | Partial / deferred | 31/35 cells complete; B128 x 1M retained as GPU-memory failure evidence; B256 calibrated through batch 1,000; larger B256 batches deferred and are not V1 blockers |
| M8 | Automate experiment pipeline | Complete | Resumable `scripts/run_study.py` |
| M9 | Construct master performance terrain | Complete | CPU, parallel CPU, direct GPU, GPU surrogates |
| M10 | Define Pareto analyses | Complete | Batch-conditioned and conventional frontiers |
| M11 | Produce publication-quality figures | Complete | Direct, accuracy, master-terrain, and Pareto figures are reproducible pipeline outputs |
| M12 | Interpret engineering results | Complete | Automated crossover, accuracy-cost, dominance, empirical scaling, feasibility boundary, and V1 conclusion generation |
| M13 | Perform final reproducibility run | Complete | Retained evidence regenerated all V1 scientific results through M12 analysis; processed data were unchanged and regenerated figures were pixel-identical |
| M14 | Generate V1 report | Complete | Generated Markdown report consumes authoritative M12 analysis and reproducible publication figures |
| M15 | Freeze V1 release | Pending | Clean repository, commit, tag |

## Frontier terminology

### Batch-conditioned best-exact frontier

For each fixed batch size, select the fastest exact implementation from the
available exact methods.

This frontier answers:

> Given this workload size, which exact implementation should be used?

It captures the observed transition among scalar CPU, parallel CPU, and direct
GPU execution as batch size increases.

### Conventional surrogate Pareto frontier

Compute nondominance among surrogate measurements in the two objectives

- nanoseconds per evaluation, and
- Linf absolute error.

Batch size is not treated as a constraint.

This frontier answers:

> Among measured surrogate operating points, which choices are not dominated
> simultaneously in speed and accuracy?

### Conventional global Pareto frontier

Compute the same two-objective nondominance relation over all feasible methods,
including exact and approximate computation.

This frontier answers:

> If batch size is ignored as a constraint, which measured operating points are
> globally nondominated in speed and error?

Because exact methods have zero approximation error, a sufficiently fast exact
method may dominate the entire surrogate family. That is a valid engineering
result rather than a failure of the surrogate experiment.

## M12 — Engineering interpretation

The analysis must address the following before V1 is frozen.

### M12.1 Exact crossover analysis

Determine the workload regions in which the preferred exact implementation is:

- scalar CPU,
- parallel CPU,
- direct GPU resident, or
- direct GPU end-to-end.

Report crossover locations and speedup factors.

### M12.2 Surrogate cost-of-accuracy analysis

For each surrogate budget, quantify:

- RMSE,
- p99 absolute error,
- Linf absolute error,
- execution cost,
- throughput, and
- cost relative to the best exact method at the same batch size.

### M12.3 Dominance analysis

Determine whether any surrogate measurement dominates an appropriate exact
baseline and whether any surrogate remains on the conventional global Pareto
frontier.

Distinguish conventional global dominance from comparisons conditioned on
batch size.

### M12.4 DAG scaling analysis

Relate surrogate budget and accuracy to native CPWA-ReLU representation size,
including:

- affine nodes,
- min nodes,
- max nodes,
- total nodes,
- artifact size,
- lowering/compilation cost where available, and
- steady-state execution cost.

The purpose is to identify why increasing approximation accuracy does or does
not remain computationally attractive.

### M12.5 Feasibility boundary

Treat failed or impractical calibration combinations as experimental evidence.

Record each requested budget × batch combination using explicit measurement
semantics:

- `complete` — a valid retained measurement exists;
- `resource_failure` — the experiment was attempted and failed because of a
  documented hardware/resource limit;
- `failed` — the experiment was attempted but failed for another or
  unclassified reason;
- `deferred` — the experiment was deliberately not run because existing
  evidence is sufficient for the current V1 conclusion; or
- `pending` — the experiment remains unresolved.

A failure must retain enough evidence to explain its classification. Deferred
measurements are measurement decisions, not failed experiments.

### Automated M12 evidence

`scripts/analyze_study.py` derives the reusable M12 evidence tables from retained
measurements. It writes exact crossover, surrogate accuracy/cost, dominance,
DAG-scaling, empirical scaling, feasibility, and a machine-readable
`study-summary.json`. The summary records representation and accuracy/cost
scaling exponents, incremental tradeoffs, exact-method transitions, resource
boundaries, dominance conclusions, and measurement sufficiency. Empirical
power-law fits describe the measured range and are not asserted as asymptotic
complexity results.

For ThermoGPU V1, all 31 measured surrogate operating points are dominated by
the best exact implementation at the same batch size and no surrogate point is
globally Pareto-optimal. The B128 × 1,000,000 calibration establishes a
hardware-conditioned GPU-memory resource boundary. The three larger B256
measurements are deliberately deferred because they cannot change the current
V1 dominance conclusion. There are no unresolved pending cells.

The processor accepts explicit input/output paths so the same analysis contract
can be reused by later engineering case studies rather than reimplemented per
model.

### M12.6 V1 measurement decision

Before running additional expensive experiments, determine whether the missing
measurement could change any V1 conclusion.

Additional measurements are required only when they can plausibly:

- change an implementation crossover,
- change a Pareto frontier,
- change a surrogate accuracy/cost conclusion,
- establish an unknown feasibility boundary, or
- resolve an unexplained numerical or performance anomaly.

## M13 — Reproducibility result

The V1 reproduction run was performed from retained experimental evidence using
the pipeline's evidence-only reproduction mode. Evidence-producing stages were
preserved rather than rerun, while derived processing, plotting, master-Pareto,
and engineering-analysis stages were regenerated.

The regenerated processed scientific results produced no numerical or textual
differences from the retained V1 results. The direct-throughput and
surrogate-accuracy PNG files were not byte-identical because of PNG
serialization differences, but decoded image dimensions and RGBA pixels were
identical to the committed figures.

The reproduced engineering conclusion remained:

- 7 exact batch operating points analyzed;
- 31 surrogate measurements analyzed;
- 31/31 measured surrogate points dominated by the best exact method at the
  same batch size; and
- 0 surrogate points on the conventional global Pareto frontier.

The reproduction run does not rerun expensive experimental measurements merely
to reproduce derived scientific conclusions.

## M14 — V1 report result

The V1 report is generated from the authoritative processed evidence and M12
`study-summary.json`; it no longer depends on the legacy surrogate-performance
aggregate. The reproduction pipeline now regenerates the master performance
terrain and Pareto figures before regenerating the report.

The report records the frozen design, exact-method crossover, surrogate accuracy
and representation scaling, dominance and Pareto results, feasibility boundary,
measurement sufficiency, engineering interpretation, limitations, and the
evidence-only reproduction command.

## V1 exit criteria

V1 may be frozen when:

1. direct exact methods are validated and benchmarked;
2. surrogate accuracy is validated on the frozen common validation set;
3. all measurements material to the conclusions are complete, have a
   documented failure classification, or are explicitly deferred with a
   scientific-sufficiency justification;
4. batch-conditioned and conventional Pareto analyses are reproducible;
5. crossover, dominance, DAG-scaling, and feasibility conclusions are
   documented;
6. publication figures regenerate from retained evidence;
7. the final report regenerates from retained evidence;
8. the automated pipeline and tests pass from a clean checkout; and
9. repository provenance is sufficient to reproduce the reported results.

The completion criterion is scientific sufficiency, not rectangular coverage
of every possible experiment combination.
