# Cross-Stack SpMV Comparison on B70 — Ginkgo 3.2× Ahead of PETSc aijkokkos

## Summary

A standalone SpMV microbenchmark on the same B70 hardware, same SYCL
runtime, same 1M × 1M Poisson 5-point stencil shows Ginkgo's `dpcpp`
backend (this repo's path) at **3.2× the throughput of PETSc's
`MATAIJKOKKOS`** (the [sister
repo's](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70)
path).

| Stack | ms / iter | GFLOPS | Effective BW |
|---|---|---|---|
| PETSc aijkokkos (β5h2 Release, oneAPI 2025.3) | 0.287 | 35 | 418 GB/s |
| **Ginkgo dpcpp (`/opt/ginkgo`, oneAPI 2026.0)** | **0.089** | **112** | **1340 GB/s\*** |

\* Cache-resident `x` vector (8 MB) fits in B70's 12 MB L2. The
arithmetic effective-BW exceeds physical peak (608 GB/s); the
meaningful number is the wall-time per iter.

## Cross-Reference

The same benchmark methodology, source, and matrix are documented in
the sister repo:

- [Sister Finding 23 — B70 hardware functional](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70/blob/main/findings/23_b70_hardware_functional_amg_wall_is_software.md)
- [Sister Finding 24 — PETSc aijkokkos at 79 % Triad](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70/blob/main/findings/24_petsc_aijkokkos_spmv_79_percent_triad.md)
- [Sister Finding 25 — the same comparison from PETSc's perspective](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70/blob/main/findings/25_ginkgo_3x_faster_microbench.md)

## Limitations — Read Before Drawing Conclusions

This is a **microbenchmark**, not a production result. Three precise
caveats:

1. **Cache-resident `x`.** With 1M unknowns × 8 bytes = 8 MB, `x` lives
   in B70's L2 across iterations. For larger systems (≥ 4M unknowns)
   or multi-RHS workloads, this advantage shrinks toward the physical
   VRAM-BW ceiling (~530 GB/s sustained Triad).
2. **Hand-tuned vs default kernel.** Ginkgo's `dpcpp` CSR-SpMV kernel
   has been actively maintained against DG2/PVC since 2023. PETSc's
   `aijkokkos` dispatches to KokkosKernels' default CSR-SpMV. The 3.2×
   reflects "tuned vs default" more than "Ginkgo vs PETSc generally".
3. **SpMV-only.** Says nothing about preconditioner setup, multigrid
   construction, BJ-block-Jacobi convergence, or solver stability.
   Those determine the production CFD runtime — and on B70 in May 2026
   they are blocked for both stacks (this repo's findings 02-22 plus
   the sister repo's findings 19-22).

## What This Means for This Repo

If this repo's solver-stability blockers (finding
[02](02_bj_blocksize_int_underflow.md) BJ underflow, finding
[05](05_sycl_preconditioner_status.md) SYCL triangular-solve gap,
finding [22](22_vram_pressure_gmres_oom.md) GMRES VRAM pressure) get
fixed upstream, the resulting Ginkgo + OGL stack on B70 has a
plausible path to outperforming the petsc4Foam stack — assuming the
SpMV advantage carries through to full Krylov solver workloads.

This is **motivation**, not validation. The performance verdict is
gated on the convergence/stability work tracked in the rest of this
repo's findings, not on this microbenchmark.

## What This Means for the Sister Repo

Their corresponding [Finding 25](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70/blob/main/findings/25_ginkgo_3x_faster_microbench.md)
documents the same number from the petsc4Foam side. They reach the
honest conclusion that for SpMV-bound paths (CG + Jacobi or
Chebyshev), a Ginkgo path is the more performant target on B70 — *if
this repo's work converges*.

The cross-link is symmetric: neither repo can claim production
victory without the other. Both teams (us + the petsc4Foam pioneer)
need their respective software-stack maturity to land before the May
2026 hardware verdict moves.

## Setup

Same diagnostic harness as the sister repo. Reproducer source under
`/home/heiko/diag/` on the Tavea-Station workstation (not committed —
diagnostic scratch space).

```
Hardware:    Intel Arc Pro B70 Pro (BMG-G31, device 0xe223)
Driver:      Linux 7.0.0 + Mesa 26.05 + intel-compute-runtime 26.05
oneAPI:      2026.0 (default symlink) for Ginkgo
             2025.3.3 for PETSc β5h2 (in sister diag harness)
Matrix:      1000 × 1000 5-point Poisson, 1M unknowns, ~5M nnz
Bench:       1000 × MatMult after 10 warmup, single timer span
```

## Evidence

- `logs/diag-2026-05-10/test3a_petsc.log.gz` — PETSc number for
  cross-reference
- `logs/diag-2026-05-10/test3b_ginkgo.log.gz` — Ginkgo number

## Status / Resolution

**Cross-stack microbenchmark logged.** Production interpretation
remains gated on the solver-stability and OGL-integration work tracked
in this repo's other findings.

## Related (this repo)

- [02 — BJ blocksize integer underflow](02_bj_blocksize_int_underflow.md)
- [05 — SYCL preconditioner status](05_sycl_preconditioner_status.md)
- [22 — GMRES VRAM pressure](22_vram_pressure_gmres_oom.md)

## Related (sister repo)

- Findings 23–26 on PETSc/Kokkos side, full diagnostic context
