# ThermoGPU Case Study Specification

## 1. Objective

Determine the numerical-accuracy and computational-cost tradeoffs among direct
CPU, parallel CPU, CUDA GPU, and adaptively constructed CPWA surrogate
evaluation of an engineering thermodynamic quantity.

The initial quantity is the methane Peng--Robinson compressibility factor

    Z(T, P)

over a fixed two-dimensional temperature-pressure domain.

The study is intended to establish a reproducible methodology that can later
be applied to compressor-station and transient-pipe models.

---

## 2. Frozen software stack

The initial study uses:

    ThermoGPU   v1.1.0
    TDAR        v0.3.0
    CPWA-ReLU   v0.2.0

Exact commit hashes shall be recorded with the final experiment manifest.

These releases are dependencies of the experiment, not development targets.

If a correctness defect is discovered, it must be documented and fixed in the
appropriate upstream repository. New optimization research is outside the
scope of this case study.

---

## 3. Methods under comparison

### Direct physics

1. Scalar CPU Peng--Robinson evaluation
2. Parallel CPU Peng--Robinson evaluation
3. CUDA GPU Peng--Robinson evaluation

These methods evaluate the same underlying thermodynamic model using different
execution mechanisms.

### Surrogate

4. ThermoGPU oracle
5. TDAR adaptive sampling
6. Simplicial CPWA representation
7. CPWA-ReLU compilation/execution

The surrogate intentionally changes the computational strategy: expensive
direct evaluations are used during construction so that subsequent queries can
use a cheaper approximation.

Future accelerator targets, including TPU/integer execution, may be added
after the initial methodology and measurements are frozen.

---

## 4. Scientific questions

The study shall answer:

1. How do scalar CPU, parallel CPU, and CUDA throughput scale with batch size?

2. What approximation error is obtained by TDAR/CPWA surrogates at different
   sampling budgets?

3. What is the evaluation cost of each resulting surrogate?

4. What accuracy/performance tradeoff is obtained across direct and surrogate
   methods?

5. How many evaluations are required before surrogate construction cost is
   amortized?

6. Which methods are Pareto-optimal under the selected error and performance
   metrics?

---

## 5. Accuracy metrics

Surrogate predictions shall be evaluated against a frozen high-fidelity
reference set.

At minimum record:

    RMSE
    p99 absolute error
    maximum absolute error (L_inf)

Additional normalized or relative metrics may be reported where physically
meaningful, but they shall not replace the primary absolute-error measurements.

Direct implementations shall retain their existing differential-validation
evidence against the validated ThermoGPU reference.

---

## 6. Performance metrics

At minimum record:

    total elapsed time
    ns/evaluation
    evaluations/second
    batch size

For surrogate construction also record:

    number of oracle evaluations
    oracle evaluation cost
    adaptive construction cost
    representation/compiler cost
    total build cost

Compilation and one-time setup costs shall not be silently mixed with
steady-state evaluation costs.

---

## 7. Batch scaling

Target batch sizes are initially:

    1
    10
    100
    1,000
    10,000
    100,000
    1,000,000

A method is not required to produce every point if a documented implementation
or hardware limitation prevents a meaningful measurement.

Missing measurements shall be reported rather than converted into new
optimization work.

Existing validated benchmark results should be reused where their methodology
matches this specification.

---

## 8. Surrogate budgets

The initial candidate TDAR budgets are:

    16
    32
    64
    128
    256

These are experimental sampling points, not a requirement to manufacture five
successful results.

Existing TDAR measurements should be inventoried before new runs are made.

Additional budgets may be used only when needed to resolve the Pareto frontier,
not simply to increase experiment count.

---

## 9. Pareto analysis

The primary Pareto relationship is:

    approximation error <-> steady-state evaluation cost

with evaluation cost represented by ns/evaluation or its reciprocal,
evaluations/second.

A method is Pareto dominated if another measured method is both more accurate
and less computationally expensive.

Direct physics methods are expected to occupy the near-reference-accuracy end
of the comparison, while surrogate configurations may provide a family of
accuracy/performance tradeoffs.

The observed data, rather than the expected ordering, determine the frontier.

---

## 10. Amortization

Let

    C_b = surrogate construction cost
    C_s = surrogate cost per evaluation
    C_d = direct-model cost per evaluation

Then

    C_surrogate(N) = C_b + N C_s

and

    C_direct(N) = N C_d.

When C_s < C_d, the nominal break-even evaluation count is

    N* = C_b / (C_d - C_s).

Break-even counts shall be reported for meaningful direct/surrogate
comparisons.

Both steady-state performance and amortized performance must therefore be
distinguished in the final analysis.

---

## 11. Required figures

The V1 case study should produce at least:

1. batch size vs throughput;
2. approximation error vs evaluation cost Pareto plot;
3. TDAR sample budget vs approximation error;
4. total computational cost vs number of requested evaluations;
5. surrogate/direct break-even visualization.

Additional figures are optional and must support a specific scientific claim.

---

## 12. Reproducibility

Final measurements shall record:

- repository versions and commit hashes;
- compiler/runtime versions;
- Python environment where applicable;
- CPU model;
- GPU model;
- operating system;
- relevant accelerator/runtime configuration;
- benchmark command;
- random seeds where applicable;
- raw trial measurements.

Raw measurements shall be retained separately from processed tables and
figures.

Figures and reported aggregate results must be reproducible from retained raw
data.

---

## 13. Existing-data-first rule

Before running new benchmarks:

1. inventory existing ThermoGPU results;
2. inventory existing TDAR ThermoGPU results;
3. inventory existing CPWA-ReLU ThermoGPU results;
4. map those measurements onto this specification;
5. identify the minimum missing measurements.

Only the missing measurements required to answer the scientific questions
should then be generated.

---

## 14. Explicit non-goals

The ThermoGPU V1 case study does not include:

- new TDAR refinement algorithms;
- new CPWA mathematics;
- new DAG schedulers;
- new GPU register heuristics;
- further CP-SAT DAG minimization research;
- Maxwell-specific kernel tuning;
- arbitrary surrogate examples;
- compressor-station modeling;
- transient-pipe modeling;
- TPU optimization.

Interesting observations in these areas should be recorded as future work
rather than pursued inside this study.

---

## 15. Acceptance criteria

The ThermoGPU case study is complete when:

1. the software and hardware configuration is frozen and recorded;
2. direct CPU, parallel CPU, and CUDA performance evidence is assembled;
3. at least one validated TDAR -> CPWA -> CPWA-ReLU path is demonstrated;
4. surrogate accuracy is measured against the frozen reference;
5. steady-state direct/surrogate performance is compared;
6. construction cost and amortization are quantified;
7. the measured Pareto frontier is produced;
8. raw data and figure-generation procedures are retained;
9. limitations and negative results are documented; and
10. a concise engineering conclusion identifies the useful operating regimes
    of the measured approaches.

Completion does not require solving every performance anomaly discovered
during measurement.

A missing Pareto point does not create a new research project.

---

## 16. Future extensions

After V1 is frozen:

- add TPU/integer surrogate execution as an additional hardware target;
- apply the methodology to a compressor-station model;
- apply the methodology to transient pipe simulation.

These extensions should reuse the experimental framework established here
rather than redefine the ThermoGPU V1 experiment.
