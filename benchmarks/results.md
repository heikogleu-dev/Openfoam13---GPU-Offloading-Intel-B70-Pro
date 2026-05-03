# Benchmark Results

## Test Case

- **Mesh:** 34,088,573 cells (MR2 SW20 vehicle aero)
- **Solver:** simpleFoam incompressibleFluid, k-omega SST
- **Metric:** Steady-State average s/Step (Time=8,9,10 mean unless noted)
- **nNonOrthogonalCorrectors:** 2 unless noted

## Complete Performance Table

| # | Test | np | Solver | Precond | nNonOrth | maxIter | s/Step | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | 2 | GKOCG | BJ | 2 | 200 | 96.0 | First working config |
| 2 | np=8 | 8 | GKOCG | BJ | 2 | 200 | 52.0 | 1.84× vs baseline |
| 3 | maxIter cap | 2 | GKOCG | BJ | 2 | 50 | 83.0 | Per-iter overhead dominates |
| 4 | ISAI | 2 | GKOCG | ISAI | 2 | 200 | 146 (div) | Diverges, scaling -1.0 ineffective |
| 5 | IC | 2 | GKOCG | IC | 2 | — | crash | NotImplemented in Ginkgo SYCL |
| 6 | np=8 (repeat) | 8 | GKOCG | BJ | 2 | 200 | 52.0 | Reproducible |
| 7 | np=16 optimized | 16 | GKOCG | BJ | 1 | 80 | **30.6** | Best GPU (reduced quality) |
| 8 | np=16 fair | 16 | GKOCG | BJ | 2 | 200 | ~50.0 | Fair vs CPU |
| 9 | U+k+ω on GPU too | 8 | GKOBiCGStab | BJ | 2 | 50 | >200 | 22 s/k-solve overhead |
| 10 | **CPU GAMG np=8 P-only** | 8 | GAMG | DIC | 2 | 50 | **43.3** | Native Linux baseline |
| 11 | **CPU GAMG np=16 P+E** | 16 | GAMG | DIC | 2 | 50 | **35.7** | **Best overall** |
| 12 | maxBlockSize 32 | 16 | GKOCG | BJ(32) | 2 | 200 | OOM | 70 GB VRAM needed |
| 13 | maxBlockSize 16 | 8 | GKOCG | BJ(16) | 2 | 200 | OOM | SYCL workspace bug |
| 14 | maxBlockSize 8 | 8 | GKOCG | BJ(8) | 2 | 200 | OOM | even at smallest BS>1 |
| 15 | Hybrid format | 8 | GKOCG | BJ | 2 | 200 | error | "not supported in distributed mode" |
| 16 | ICT (ParICT) | 8 | GKOCG | ICT | 2 | 200 | DEVICE_LOST | GPU hardware hang |

## RAM Bandwidth (STREAM Triad)

| Threads | Config | Triad GB/s |
|---|---|---|
| 1 | Single P-Core | 38.0 |
| 8 | P-Cores only | **77.0** |
| 16 | P+E mixed | 73.8 |
| 24 | All threads | 70.5 |

> RAM bandwidth saturates at 8 P-Cores (71% of DDR5-6800 theoretical 109 GB/s).
> Adding E-Cores does NOT improve memory-bound performance — slight decrease
> due to coherence overhead.

## CPU FP64 DGEMM

| Threads | Backend | GFLOPS |
|---|---|---|
| 8 | MKL (preloaded) | 592 |
| 24 | MKL (preloaded) | 1106 |
| 1 | Reference BLAS | 6.8 |

> Ubuntu default `numpy` uses Reference BLAS (single-threaded, 87× slower
> than MKL). Always preload MKL for compute work:
> ```bash
> LD_PRELOAD=/opt/intel/oneapi/mkl/2026.0/lib/libmkl_rt.so.3 python3 ...
> ```

## GPU Bandwidth (custom SYCL benchmark)

| Metric | Value | Spec | Efficiency |
|---|---|---|---|
| VRAM D2D Triad (4 MB) | 508 GB/s | 608 GB/s | 83% |
| VRAM D2D Triad (256 MB) | 260 GB/s | 608 GB/s | 43% |
| H2D (64 MB) | 15.5 GB/s | ~50 GB/s (PCIe 5×16) | 31% |
| D2H (64 MB) | 14.4 GB/s | ~50 GB/s | 29% |
| Kernel Launch Latency | ~100 µs | ~5 µs (CUDA) | 5% |

## VRAM Usage Profile

Idle baseline: 2.27 GiB (display server on B70 Pro). With display on iGPU
expect <0.5 GiB.

| Config | VRAM under load |
|---|---|
| GPU np=8 BJ | 9.34 GiB / 32 |
| GPU np=16 BJ | 10.06 GiB / 32 |
| GPU np=8 with all 4 solvers SYCL (test 9) | 29.5 GiB / 32 (peak) |

Plenty of headroom for the BJ-only configs. The all-4-solvers-SYCL approach
nearly hit OOM and is ill-advised for this mesh size.

## CPU/GPU Time-Step Comparison Table

For "fair" config (nNonOrth=2, full convergence settings):

```
Setup phase (Time=1):
  GPU np=2:  188 s
  GPU np=8:   94 s  (parallel mesh load + JIT)
  GPU np=16:  67 s
  CPU np=8:   68 s
  CPU np=16:  56 s

Steady-state per step (Time=8-10 mean):
  GPU np=2 BJ:    96.0 s
  GPU np=8 BJ:    52.0 s
  GPU np=16 BJ:  ~50.0 s  (extrapolated from short run)
  CPU np=8 GAMG:  43.3 s
  CPU np=16 GAMG: 35.7 s  ← winner
```
