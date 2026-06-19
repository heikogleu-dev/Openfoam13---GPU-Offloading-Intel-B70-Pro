# Why a busier GPU (36% util) gives NO speedup at 34M — timing/comm/util analysis

**Date:** 2026-06-19. **Setup:** full-float (FP32) GKOCG + Multigrid(single, CG-coarse),
caching 2 unless noted, CR 26.05, B70, desktop on iGPU. Instrumentation:
`run-timing-util.sh` (fdinfo ccs/rcs/bcs cycle deltas = compute/render/copy engine util)
+ OGL verbose=1 timings. CPU-GAMG baselines: 7.1M ~9, 17.2M 22.1, 34M 35.7 s/step.

## The puzzle
GPU compute util RISES with mesh (25.9% → 29.1% → 35.7%) yet the GPU goes from
**winning** (7–17M) to **losing** (34M) vs CPU-GAMG. Why does higher util ≠ speedup?

## Step decomposition (caching 2, 3 p-solves per SIMPLE step)
| Mesh | np | s/step | CCS% | BCS% | GPU-p-solve/step | CPU-rest/step | CPU-GAMG | **GPU-p vs GAMG-p** |
|---|---|---|---|---|---|---|---|---|
| 7.1M | 8 | 7.0 | 25.9 | 13.7 | 1.9 s | 5.1 s | 9.0 | **2.03× faster** |
| 17.2M | 16 | 18.0 | 29.1 | 8.6 | 6.2 s | 11.8 s | 22.1 | **1.66× faster** |
| 34M | 16 | 37.6 | 35.7 | 6.2 | 14.6 s | 23.0 s | 35.7 | **0.87× (slower)** |

GPU-p-solve/step = 3 × `solve_multi_gpu`. CPU-rest = step − GPU-p (same in both runs:
U/k/omega DILU + FVM assembly + flux + MPI). GAMG-p = GAMG-step − CPU-rest.

## Answer — three points
1. **Util% is a RATE, not throughput.** CCS=36% only means the GPU is busy 36% of the
   wall. It rose because the GPU's *job got disproportionately more expensive*, not
   because it delivers more. A busier GPU here is the **symptom**, not a win.
2. **The decisive metric is GPU-p-solve vs GAMG-p-solve — and the GPU edge erodes
   2.03× → 1.66× → 0.87×**, crossing below 1.0 between 17.2M and 34M. At 34M the GPU AMG
   p-solve (14.6 s) is *more expensive* than CPU-GAMG (12.7 s). Since CPU-rest (~23 s) is
   identical, that tips the whole step.
3. **GPU-AMG scales superlinear, CPU-GAMG sublinear** (clean 17.2M→34M, both np16, 2× cells):

| Component (per solve) | 17.2M | 34M | factor / 2× cells |
|---|---|---|---|
| init_precond **build** (Galerkin RAP + aggregation) | 1643 ms | 4689 ms | **2.85×** ⚠ worst |
| solve / apply (V-cycle, SpMV) | 859 ms | 1981 ms | 2.30× |
| all2all comm (host-double→device-float gather) | 226 ms | 441 ms | 1.95× |
| copy_x_back | 11 ms | 23 ms | 2.1× |
| iters | 12.0 | 13.4 | 1.12× |
| **GPU step total** | 18.0 s | 37.6 s | **2.09× (superlinear)** |
| **CPU-GAMG step** | 22.1 s | 35.7 s | **1.62× (sublinear)** |

The **AMG hierarchy build** is the main culprit (2.85×): SpGEMM Galerkin products +
aggregation scale poorly on the GPU; even amortized by caching it stays heavy at 34M.
CPU-GAMG has mature cache-friendly agglomeration with few iters (6→3) → sublinear.

## Per-solve component table (caching 2)
| Mesh | init_p reuse | init_p build | solve | all2all | copyback | solve_multi_gpu |
|---|---|---|---|---|---|---|
| 7.1M np8 | ~363 | ~670 | 267 | 46 | 5 | 647 |
| 17.2M np16 | 551 | 1643 | 859 | 226 | 11 | 2059 |
| 34M np16 | 1173 | 4689 | 1981 | 441 | 23 | 4871 |
(ms; np8→np16 inflates all2all comm ~4.9× — the 7.1M↔17.2M jump mixes mesh + np doubling;
the clean mesh-only scaling is 17.2M↔34M, both np16.)

## Caching lever at 34M (np16, MG single CG)
| caching | s/step | iters | CCS% | note |
|---|---|---|---|---|
| 2 | 37.6 | 13.4 | 35.7 | |
| 3 | 36.5 | 13.5 | 31.1 | |
| 4 | 36.5 | 13.5 | 31.4 | |
| 5 | **35.7** | 13.4 | 31.0 | = CPU-GAMG parity |
Higher caching thins the 4.69 s AMG rebuilds → ~5% gain, reaches **parity** (35.7) but
**no win**. Iters unchanged (no convergence cost). The build is only part of the GPU work
and the GPU is anyway a minority of the CPU-bound step.

## Bottom line
Two effects at 34M: (a) the step is **~64% CPU-rest** the GPU can't touch; (b) even the
GPU half loses its edge because GPU-AMG scales superlinearly while GAMG scales sublinearly.
**Real levers:** cheaper/reused AMG build (RS-coarsening, build-reuse vs rebuild),
fewer iters (4th-kind Chebyshev smoother, SpMV-only — B70-ideal), and above all the
**CPU side** (U/k/omega + assembly) which dominates at 34M. Full-float's 34M win remains
**VRAM only** (fits one card; double OOMs) — see [vram-and-mesh-scaling.md](vram-and-mesh-scaling.md),
[full-float-port-plan.md](full-float-port-plan.md), [performance-maps.md](performance-maps.md).
Repro: `findings/code/gpu-diag/run-timing-util.sh`, `sweep-fullfloat.sh`.
