# ThermoGPU Case Study Evidence Inventory

This document maps the retained V1 evidence onto the requirements of the
ThermoGPU case-study specification. It is a release inventory, not a request
for additional measurements.

Frozen stack:

- ThermoGPU v1.1.0
- TDAR v0.3.0
- CPWA-ReLU v0.2.0

| Required evidence | Retained artifact | V1 status |
|---|---|---|
| Direct CPU/OpenMP/GPU performance and scaling | `raw-results/direct/scaling-b1-10-100-1000-10000-100000-1000000.json` | Complete |
| Direct-model environment/provenance | `raw-results/environment.json`, `raw-results/pipeline-manifest.json` | Complete |
| TDAR surrogate family and simplicial CPWA export | `raw-results/surrogates/artifacts/methane-z-b{16,32,64,128,256}.{json,npz}` | Complete |
| Frozen common validation set and truth | `raw-results/surrogates/validation-points.npy`, `validation-truth.npy` | Complete |
| Surrogate numerical accuracy | `raw-results/surrogates/report.json`, `summary.csv`, `normalized.json` | Complete |
| Exact CPWA-ReLU native DAGs | `raw-results/surrogate-performance/budget-*/artifacts/*.cpwa` and `build-native.json` | Complete |
| Surrogate GPU calibration/performance | `raw-results/surrogate-performance/budget-*/calibration/` | 31 complete cells; one documented resource failure; three explicit deferrals |
| B128 × 1,000,000 feasibility boundary | `raw-results/surrogate-performance/budget-128/calibration/float32-b1000000.failed.json` | Documented GPU-memory resource failure |
| Direct/surrogate comparison and Pareto inputs | retained direct and per-cell calibration evidence above | Complete; derived reproducibly |
| Crossover, dominance, scaling, and feasibility analysis | `processed-results/` and `processed-results/study-summary.json` | Complete; derived reproducibly |
| Publication figures | `figures/` | Complete; derived reproducibly |
| Final V1 report | `reports/thermogpu-v1.md` | Complete; derived reproducibly |
| Evidence-only reproduction provenance | `raw-results/pipeline-reproduction-manifest.json` | Complete |

## Inventory conclusions

The retained evidence is sufficient for the V1 engineering conclusion. Across
the measured design, 31 of 31 surrogate operating points are dominated by the
best exact implementation at the same batch size, and no surrogate point is on
the conventional global Pareto frontier. The B128 × 1,000,000 attempt records
a GPU-memory resource boundary.

The B256 × 10,000, × 100,000, and × 1,000,000 cells are deliberately deferred,
not missing or failed measurements. Existing evidence is sufficient to show
that they are not required to establish the V1 conclusion. There are no
unclassified pending cells.

## Additional measurements

No additional measurements are required for the V1 conclusion. Future
measurements belong to a new study or revision if they can plausibly change an
implementation crossover, Pareto frontier, accuracy/cost conclusion, known
feasibility boundary, or resolve a numerical/performance anomaly.

Derived products are regenerated from retained evidence with `run_study.py
--reproduce`; expensive evidence-producing stages are not rerun merely to
reproduce the V1 analysis.
