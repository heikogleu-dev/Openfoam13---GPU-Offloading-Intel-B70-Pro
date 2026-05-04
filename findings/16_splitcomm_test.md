# Test: splitComm=false has no measurable effect on s/step

## Background

OGL's README documents:

> `splitComm: true (default)` — whether to split the communicator for
> the host and device side

We had been running with the default (`true`) throughout all benchmarks.
This finding tests whether explicitly setting `splitComm false` changes
performance.

## Setup

- 34M cells, np=8, scotch, GKOCG + BJ(maxBlockSize=1), `forceHostBuffer=true`
- 3 timesteps, steady-state mean of T=2 and T=3
- p-block: added `splitComm false;` explicitly

## Result

| Setting | s/step (T2,T3 mean) | vs Baseline |
|---|---|---|
| `splitComm true` (default, baseline) | 53.8 | — |
| `splitComm false` | 54.6 | +1.5 % (within noise) |

Iteration counts identical (200 cap hit on every solve, as expected for
BJ(1)).

## Verdict

**No measurable effect for our case.** With `forceHostBuffer=true`
already mandatory for xe (no GPU-aware MPI), the host/device communicator
split has no observable performance impact in the BJ(1)-bound regime where
the dominant cost is the MPI Allreduce wait between iterations
(see [profiling/bottleneck_analysis.md](../profiling/bottleneck_analysis.md)).

`splitComm` is likely meaningful for setups with GPU-aware MPI where
host-side and device-side messages can be routed differently to optimise
PCIe utilisation. With our forced-host-buffer pattern, every message
traverses the host side anyway.

## Implication

Stick with the default `splitComm true`. No tuning gain available here
without first solving the GPU-aware-MPI problem.
