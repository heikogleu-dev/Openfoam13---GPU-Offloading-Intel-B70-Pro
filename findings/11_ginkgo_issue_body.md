<!--
Prepared body for opening an issue at https://github.com/ginkgo-project/ginkgo/issues/new
Use:
  gh issue create --repo ginkgo-project/ginkgo \
    --title "SYCL/Battlemage: BJ maxBlockSize>1 OOM + IC/ICT NotImplemented/DEVICE_LOST on Intel Arc Pro B70 Pro" \
    --body-file findings/11_ginkgo_issue_body.md
-->

## Summary

Comprehensive testing of Ginkgo 1.10 SYCL preconditioners on Intel Arc Pro
B70 Pro (Battlemage G31, 32 GB) reveals critical issues for CFD use cases.

## Hardware (performing well — software is the bottleneck)

- GPU: Intel Arc Pro B70 Pro (0xe223, 32 GB GDDR6, Ubuntu 26.04)
- FP64: 1335 GFLOPS (93% of spec) ✅
- VRAM: 530 GB/s sustained (87% of spec) ✅
- oneAPI 2026.0.0, Compute Runtime 26.05.37020

## Critical Issue 1: BJ maxBlockSize > 1 → `gko::AllocationError` (OOM)

Any `maxBlockSize > 1` causes immediate OOM during `generate()`.
Confirmed on: `maxBlockSize` = 2, 4, 8, 16, 32 — all fail identically.
Only `maxBlockSize = 1` (diagonal scaling) works.

Test configuration: 34M cells, 8 MPI ranks (4.25M cells/rank, ~270 MB/rank)
dmesg: `xe VM worker error: -12 (ENOMEM)`

Suspected cause: SYCL BJ generate workspace is O(N × BS²) instead of O(N × BS).
This makes `maxBlockSize > 1` unusable on any reasonably-sized mesh.

## Critical Issue 2: IC → `gko::NotImplemented`

`sparselib_ic` not implemented for SYCL backend:
```
dpcpp/factorization/ic_kernels.dp.cpp:21: sparselib_ic is not implemented
```

## Critical Issue 3: ICT → DEVICE_LOST (GPU hardware hang)

ICT causes immediate GPU hang requiring driver reset.
```
sycl::_V1::exception: level_zero backend failed with error: 20
(UR_RESULT_ERROR_DEVICE_LOST)
```

## Critical Issue 4: Multigrid → OOM in PGM coarsening

PGM coarsening crashes in `generate_local()` → `Csr::clone()` → `raw_copy_to()`
with DEVICE_LOST (underlying ENOMEM). The one solve that completes diverges
(Final residual > Initial residual).

Stack: `gko::multigrid::Pgm::generate_local()` → `gko::matrix::Csr::apply_impl()`
→ `DpcppExecutor::raw_copy_to()` → DEVICE_LOST

## Impact on CFD

With only `BJ maxBlockSize=1` available (point-Jacobi = diagonal scaling),
the pressure solver never converges for 34M-cell meshes — always hits the
200-iteration cap. CPU GAMG (algebraic multigrid) converges in 5-10 iterations.
GPU is 1.5× slower than CPU under fair comparison.

## Full Documentation

https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro

Findings 02, 05, 08 contain stack traces and reproduction steps.

## Question

Is `BJ maxBlockSize > 1` OOM on SYCL a known issue?
Is there a fix or workaround planned for Ginkgo 2.0?
