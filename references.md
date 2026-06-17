# References

## Primary Papers

- **Olenik et al. 2024** — *Towards a platform-portable linear algebra
  backend for OpenFOAM*, Meccanica
  [doi:10.1007/s11012-024-01806-1](https://doi.org/10.1007/s11012-024-01806-1)
  → Defines the OGL design and KIT recommendation
  *"2× MPI subdomains per GPU"*. We tested with `ranksPerGPU 8`
  (single GPU, 8 ranks) per this guidance.

- **Tsai et al. 2023** — *Providing performance portable numerics for
  Intel GPUs*, Wiley CCPE
  [doi:10.1002/cpe.7400](https://doi.org/10.1002/cpe.7400)
  → Documents `ParIC` / `ParILU` / `ParICT` / `ISAI` work on DPC++.
  **Earlier versions of this repo claimed a discrepancy with the paper —
  that was wrong.** Per Ginkgo team feedback (issue #2013), `ParIc/ParIlu`
  factorization *does* work on SYCL. The gap we hit on Battlemage was on
  the *apply* side: `lower_trs` / `upper_trs` kernels were missing in
  `dpcpp/solver/` — **closed by [PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
  merged 2026-06-02** (oneMKL trsm path, min oneAPI 2024.1). `ParIct::add_candidates`
  SIGABRT and classic `Ic`/`Ilu` (sparselib-based) absence remain. See
  [findings/05](findings/05_sycl_preconditioner_status.md) for the
  corrected mapping + status.

- **Anzt et al. 2022** — *Ginkgo: A Modern Linear Operator Algebra
  Framework for High Performance Computing*, ACM TOMS
  [doi:10.1145/3480935](https://doi.org/10.1145/3480935)
  → Architecture / executor model that OGL builds on.

## OGL / Ginkgo Upstream

- [hpsim/OGL](https://github.com/hpsim/OGL) — OpenFOAM Ginkgo Layer (GPU plugin)
  - [findings/10 issue body (ready to file)](findings/10_ginkgo2_api_breaks.md#upstream-issue-body-ready-to-file)
- [ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo)
  - [findings/11 issue body (ready to file)](findings/11_ginkgo_issue_body.md)
  - [Issue #2015](https://github.com/ginkgo-project/ginkgo/issues/2015) — open, our reports on `lower_trs` / `ParIct` / ISAI int32
  - [PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023) — merged 2026-06-02, closes `lower_trs`/`upper_trs` SYCL gap via oneMKL trsm
- [intel/compute-runtime](https://github.com/intel/compute-runtime)
  - [Issue #922 (GSD-12696)](https://github.com/intel/compute-runtime/issues/922)
    — our report: multi-rank Level Zero `resource_info.cpp:15` abort,
    CR 26.05 → 26.14 regression on BMG-G31 / Ubuntu 26.04. Still open,
    no Intel response, no fix. Persists through CR 26.18 and kernel
    7.0.0-22 — see [findings/29](findings/29_cr_26.18_root_cause_pure_l0_multiprocess_abort.md)
    for the pure-Level-Zero minimal reproducer + follow-up
    (`logs/cr26.18-root-cause/issue_922_followup_pure_l0_repro.md`).

## Related Battlemage Pioneer Work

- **PMZFX/intel-arc-pro-b70-benchmarks**
  https://github.com/PMZFX/intel-arc-pro-b70-benchmarks
  → Independent B70 Pro pioneer for LLM inference. Upstreamed Q8_0 SYCL
  fix (PRs #21527 / #21638 in `llama.cpp`), achieving 3.1× speedup.
  → Validates our broader observation that Battlemage SYCL kernels need
  targeted fixes per workload — not a generic driver/compiler issue.

- **llama.cpp Issue #21517**
  https://github.com/ggml-org/llama.cpp/issues/21517
  → "Update from CR 26.05 to 26.09 did not improve performance — issue is
  in kernel code, not driver." Same pattern as our
  [findings/13](findings/13_stack_update_zeinit_race.md): driver updates
  alone do not solve the per-workload software-stack problems.

## Phoronix Hardware Reviews

- [Intel Arc Pro B70 Linux Benchmarks (Phoronix)](https://www.phoronix.com/review/intel-arc-pro-b70-linux)
  → Reference benchmarks on the same hardware for non-CFD workloads
  (rendering, video, ML inference). Useful for hardware sanity-check
  comparisons.

## Related Hardware/Software Documentation

- [OGL/Ginkgo recommended fvSolution patterns](https://github.com/hpsim/OGL/blob/dev/README.md)
  → Source for SPD-preconditioner `scaling -1.0` requirement we tested
  in [findings/15](findings/15_scaling_for_spd_preconditioners.md).
- [Intel Compute Runtime release notes](https://github.com/intel/compute-runtime/releases)
- [Ginkgo release notes](https://github.com/ginkgo-project/ginkgo/releases)
- [oneAPI Base Toolkit notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-toolkit-release-notes.html)


---

## Hardware Diagnostic Run — 2026-05-10

Standalone cross-stack SpMV/CG diagnostic on Intel Arc Pro B70 (BMG-G31),
Ubuntu 26.04 LTS, oneAPI 2025.3.3 / 2026.0, comparing oneMKL Sparse,
PETSc `aijkokkos`, and Ginkgo dpcpp on an identical 1M-row Poisson 5-point
reference matrix (4.996M nnz).

**Method.** Generator `gen_matrix.cpp` writes a 1000×1000 5-point Poisson
matrix in MatrixMarket format. Three test harnesses load the matrix and
run 1000 SpMV iterations after 10 warm-up calls. Timing brackets the
inner loop only; CG-loop number includes vector ops + sync per iteration.

**Hardware:** Intel Arc Pro B70, 32 GB GDDR6, BMG-G31 (device `0xe223`).
**Software:** oneAPI 2025.3.3 for PETSc β5h2, oneAPI 2026.0 for Ginkgo
(`/opt/ginkgo` linked against `libsycl.so.9`).

**Results.**

| Stack | ms/iter | Effective BW |
|---|---|---|
| oneMKL Sparse CG (full loop) | 0.741 | 161 GB/s |
| PETSc aijkokkos (pure SpMV) | 0.287 | 418 GB/s (79 % Triad) |
| Ginkgo dpcpp (pure SpMV) | 0.089 | 1340 GB/s\* |

\* Cache-resident `x` (8 MB fits in B70 L2 ≈ 12 MB). Reported BW is
arithmetic; physical peak is 608 GB/s.

**Caveat.** SpMV-only microbenchmark. The Ginkgo number reflects cache
effects that shrink for larger systems. Diagnostic value: confirms B70
hardware functional for sparse linear algebra; the AMG wall in the
sister repo is a software bug, not a hardware limitation.

**Logs:** `logs/diag-2026-05-10/` (gzipped).

**Cross-stack interpretation:** see findings 23-26 (PETSc repo) and
finding 23 (Ginkgo repo) for the symmetric write-up.

