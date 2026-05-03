# Ginkgo 1.10 SYCL Preconditioner Support Matrix

Tested on Intel Arc Pro B70 Pro, Ubuntu 26.04, Ginkgo 1.10 (OGL-bundled,
commit 061ccc3a from `ogl_0600_gko190` tag).

## Test Configuration

- 34M-cell vehicle aero case
- np=8 (4.25M cells/rank)
- nNonOrthogonalCorrectors=2
- ONEAPI_DEVICE_SELECTOR=level_zero:0
- forceHostBuffer=true (no GPU-aware MPI)

## Support Matrix

| Preconditioner | OGL String | Ginkgo Class | SYCL Status | Notes |
|---|---|---|---|---|
| Block Jacobi (BS=1) | `BJ` | `Jacobi` | ✅ Stable | Point-Jacobi, only working option |
| Block Jacobi (BS>1) | `BJ` + maxBlockSize | `Jacobi` | ❌ OOM | SYCL workspace bug |
| ISAI | `ISAI` | `Isai` | ✅ Runs | Diverges for pressure system |
| Generalized ISAI | `GISAI` | `Isai` (variant) | not tested | likely similar to ISAI |
| Incomplete Cholesky | `IC` | `Ic` + `sparselib_ic` | ❌ NotImplemented | sparselib_ic not in SYCL |
| Incomplete Cholesky T | `ICT` | `Ic` + `ParIct` | ❌ DEVICE_LOST | GPU hardware hang during generate |
| Incomplete LU | `ILU` | `Ilu` + `Lu` | not tested | likely NotImplemented |
| Incomplete LU T | `ILUT` | `Ilu` + `ParIlut` | not tested | hardware hang risk |
| Iterative Refined ILU | `IRILU` | `Ilu` + `ParIlut` | not tested | hardware hang risk |
| Multigrid | `Multigrid` | `Multigrid::Pgm` | ⚠️ Listed experimental | Not tested due to time |

## Matrix Format Support

| Format | OGL Distributed Mode | Notes |
|---|---|---|
| `Csr` | ✅ Supported | Default; works |
| `Coo` | ✅ Supported | Listed in source |
| `Ell` | ✅ Supported | Listed in source |
| `Hybrid` | ❌ Not supported | "Matrix format Hybrid not supported. Supported formats are: Ell, Csr, and Coo." (Distributed.cpp:742) |

OGL's README mentions Hybrid as a supported matrix format, but it's
explicitly rejected in the distributed-mode code path — likely only
single-rank works.

## Implication for Production CFD

For 34M-cell vehicle aero on B70 Pro with current OGL+Ginkgo 1.10 stack,
the **only stable, performance-viable preconditioner is point-Jacobi**
(`BJ` with `maxBlockSize 1`).

This severely caps achievable convergence rate vs Foundation OF13's GAMG
(algebraic multigrid). See [conclusions](../conclusions.md) for the
overall impact.

## Hopeful Future

Ginkgo 2.0 develop (`/opt/ginkgo` in our setup) may have:
- Optimized SYCL BJ generate (no OOM at maxBlockSize > 1)
- IC/ILU SYCL kernels filled in
- More robust ICT/ILUT (no DEVICE_LOST)

OGL would need to be rebuilt against external Ginkgo 2.0 (option
`OGL_USE_EXTERNAL_GINKGO=TRUE`) — untested in our session due to API
break risk.
