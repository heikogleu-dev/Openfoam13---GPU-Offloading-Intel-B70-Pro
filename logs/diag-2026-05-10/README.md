# Cross-Stack SpMV Diagnostic — 2026-05-10

Diagnostic logs from the standalone cross-stack SpMV comparison on B70.
Methodology and the PETSc-side finding are documented in the sister
repo: https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70

| File | Stack | ms / iter | Effective BW |
|---|---|---|---|
| `test3a_petsc.log.gz` | PETSc aijkokkos (β5h2 Release, oneAPI 2025.3) | 0.287 | 418 GB/s |
| `test3b_ginkgo.log.gz` | Ginkgo dpcpp (`/opt/ginkgo`, oneAPI 2026.0) | 0.089 | 1340 GB/s\* |

\* Cache-resident `x` (8 MB fits in B70's 12 MB L2). Reported BW is
arithmetic; physical peak is 608 GB/s.

→ Microbenchmark only, not a production result. See finding 23.
