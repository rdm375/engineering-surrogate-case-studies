# Engineering Synthesis

## Central result

When the computational complexity of the underlying EOS increases while surrogate input dimension remains fixed at `(T, P)`, the economic value of the hardware-executable surrogate increases substantially.

The result is workload-dependent. At batch 1, the best exact implementation dominates every measured surrogate for C1-C5. By batch 100, at least one surrogate is faster than the best exact implementation for every mixture.

## First measured crossover

| Mixture | Batch | Budget | RMSE | Speedup |
|---|---:|---:|---:|---:|
| C1 | 100 | 96 | 0.000341042 | 1.14x |
| C2 | 100 | 96 | 0.000465626 | 1.42x |
| C3 | 100 | 96 | 0.00056098 | 1.20x |
| C4 | 10 | 16 | 0.00715352 | 1.03x |
| C5 | 10 | 24 | 0.00297182 | 1.02x |

C4 and C5 already admit a narrowly faster surrogate at batch 10, although only at relatively loose measured accuracy. C1-C3 first cross over at batch 100.

## Throughput regime

At batch 1,000,000:

| Mixture | Best exact ns/eval | Most accurate faster RMSE | Speedup at that accuracy | Maximum measured speedup |
|---|---:|---:|---:|---:|
| C1 | 8.349 | 0.000341042 | 2.52x | 45.35x |
| C2 | 12.098 | 0.000313377 | 1.53x | 69.86x |
| C3 | 16.664 | 0.000388704 | 1.97x | 99.44x |
| C4 | 21.941 | 0.000354737 | 5.23x | 118.67x |
| C5 | 28.070 | 0.000359343 | 2.05x | 136.06x |

The maximum measured speedup rises from 45.35x for C1 to 136.06x for C5. More importantly, the higher-component mixtures retain substantial speedups at much tighter accuracy than the fastest low-budget surrogate.

## Why mixture complexity changes the economics

The direct Peng-Robinson calculation becomes more expensive as additional mixture components are introduced. The surrogate, however, continues to approximate a two-dimensional mapping from `(T, P)` to `Z`. Its evaluation cost is therefore governed primarily by the complexity of the synthesized CPWA function rather than by the component count of the original EOS.

The measured results consequently separate physical-model complexity from surrogate input dimension: increasing the former makes direct physics more expensive without causing a corresponding growth in the dimensionality of the surrogate problem.

## High-accuracy hardware resource transition

B96 to B128 produces a consistent hardware transition across all five mixtures. Every B96 kernel is already using 255 registers. B128 retains that register count but sharply increases local-memory spilling:

| Mixture | Spill bytes B96 | Spill bytes B128 | Stack B96 | Stack B128 |
|---|---:|---:|---:|---:|
| C1 | 8 | 676 | 8 | 680 |
| C2 | 0 | 564 | 0 | 568 |
| C3 | 48 | 420 | 48 | 424 |
| C4 | 0 | 124 | 0 | 128 |
| C5 | 156 | 724 | 160 | 728 |

Thus additional approximation accuracy is not free even when the input dimension remains fixed. At sufficiently high CPWA complexity, the generated kernel encounters a hardware resource boundary. The observed spilling is strongly associated with the B128 execution-cost increase, although this study does not establish spilling as the sole causal mechanism.

## Engineering interpretation

There are therefore two distinct crossovers. The first is between direct physics and approximation: enough repeated work is required to amortize the surrogate's GPU execution overhead. The second occurs within the surrogate family itself: increasing approximation complexity eventually encounters GPU resource pressure, making the next increment of accuracy disproportionately expensive.

The appropriate surrogate is consequently not simply the most accurate one available. It is an operating-point decision involving required accuracy, workload size, physical-model complexity, and generated-kernel resource cost.

## Scope and limitations

- Composition is fixed for each mixture; the surrogate input space is only (T,P).
- The distinct C1-C5 family changes both component count and species identity.
- Binary interaction coefficients are zero.
- The surrogate predicts Z, while the exact EOS path computes richer thermodynamic output.
- Accuracy thresholds are selected only from measured TDAR budgets; no interpolation is used.
- Direct and surrogate resident-GPU timing harnesses have comparable intent but are not identical.
- Results represent one measured hardware/software environment.
- Register spilling is strongly associated with the B128 slowdown but is not established here as the sole causal mechanism.

These conclusions apply to the measured fixed-composition, two-dimensional `(T, P)` problem and should not be generalized directly to variable-composition EOS surrogates.
