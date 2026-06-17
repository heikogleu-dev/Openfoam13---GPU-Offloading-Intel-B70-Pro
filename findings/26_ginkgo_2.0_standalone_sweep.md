# Finding 26: Ginkgo 2.0 Standalone SYCL Preconditioner Sweep — Three Bugs Are FIXED

## TL;DR

A standalone C++ test program against the installed Ginkgo 2.0 SYCL
backend on BMG-G31 (no OGL, no OpenFOAM, no MPI multi-rank — bypasses
the [CR 26.18 multi-rank gate](25_cr_26.18_multirank_pthread_race.md))
confirms that **three previously documented Ginkgo SYCL bugs are fixed
in Ginkgo 2.0**:

| Bug | 1.10 / 1.11 status | 2.0 status |
|---|---|---|
| `dpcpp::jacobi::find_blocks` `size_t` underflow ([Finding 02](02_bj_blocksize_int_underflow.md)) | crash, BJ(>1) unusable | ✅ **FIXED** — BJ(2) runs up to 9M rows |
| `dpcpp::par_ict_factorization::add_candidates` SIGABRT ([Finding 05](05_sycl_preconditioner_status.md)) | SIGABRT, ICT unusable | ✅ **FIXED** — ICT runs up to 4M rows |
| `dpcpp/solver/lower_trs_kernels.dp.cpp:43: generate is not implemented` ([Finding 05](05_sycl_preconditioner_status.md)) | NotImplemented, ILU/IC unusable | ✅ **FIXED** by [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023) — ILU runs up to 9M rows |

The only remaining algorithm-level limit observed in this sweep is the
**SYCL `int32` index overflow** in ICT for problems >4M rows
(same root cause as [Finding 15](15_scaling_for_spd_preconditioners.md)
for ISAI sparsityPower=3) — a generic constraint of the dpcpp backend's
default kernel-launch sizing, with a documented compile-flag workaround
(`-fno-sycl-id-queries-fit-in-int`).

## Why standalone

[Finding 25](25_cr_26.18_multirank_pthread_race.md) showed that CR 26.18
breaks multi-rank OGL on BMG-G31. The morning's planned A1-A3 retests
(BJ(2)/ICT/ISAI on Ginkgo 2.0) were blocked at the OGL multi-rank gate
without ever reaching the Ginkgo algorithm layer.

This finding sidesteps that gate completely: a single-process C++
program linking directly against the installed `libginkgo*.so.2.0.0`,
using `DpcppExecutor` directly. The same algorithms that crashed in
1.10/1.11 are tested in isolation here.

## Setup

- Test program: `ginkgo_precond_sweep.cpp` (preserved alongside this
  finding, also at `/home/heiko/ginkgo-standalone-tests/`)
- Built against the OGL-installed Ginkgo 2.0 develop fetched via CPM
  during the [PR #168 patched build (Finding 24)](24_pr168_patched_ilu_first_test.md)
- Matrix: 2D Poisson 5-point on an N×N grid, N² rows, ~5N² nnz
- Executor: `gko::DpcppExecutor` (no MPI, single process)
- Solver: `gko::solver::Cg` with each preconditioner in turn
- Stop criterion: max 200 iterations or residual reduction 1e-6

## Results

All results in milliseconds (total round-trip from preconditioner
factory build through CG solve), with **drei** vorher kaputte
preconditioner now successfully completing the solve at every scale up
to the noted limit:

| Preconditioner | N=100 (10k rows) | N=500 (250k) | N=1000 (1M) | N=2000 (4M) | N=3000 (9M) |
|---|---|---|---|---|---|
| BJ(maxBlockSize=1) | 198 ms | 47 | 65 | 186 | 391 ms |
| **BJ(maxBlockSize=2)** ← Finding 02 bug | 199 ms | 119 | 435 | 1753 | **3909 ms ✅** |
| **ILU (ParIlu + lower_trs apply)** ← lower_trs gap | 789 ms | 1418 | 2854 | 6463 | **12201 ms ✅** |
| **ICT (ParIct + Ic apply)** ← Finding 05 bug | 1566 ms | 7776 | 18083 | 39008 | ❌ int32 overflow |
| ISAI sparsityPower=1 | 179 ms | 60 | 183 | 670 | 1545 ms |

### ICT failure at 9M rows is NOT the `add_candidates` bug

```
ICT (ParIct factor + Ic apply):
    Exception: Provided range and/or offset does not fit in int.
               Pass `-fno-sycl-id-queries-fit-in-int' to remove this limit.
```

This is the **SYCL `int32` index overflow** documented for ISAI
sparsityPower=3 in [Finding 15](15_scaling_for_spd_preconditioners.md):
the dpcpp backend's `sycl::range`/`sycl::id` default to `int32` for
kernel launch sizing. Ginkgo can be built with
`-fno-sycl-id-queries-fit-in-int` to widen to `int64` at runtime cost —
not done in the OGL-shipped Ginkgo 2.0 build.

The `add_candidates` SIGABRT that blocked ICT in 1.10 / 1.11 (Finding 05)
is **not** reproduced at any scale tested here.

## Build / configure notes

Two small environmental notes for anyone reproducing:

1. The OGL-installed Ginkgo's `GinkgoConfig.cmake` had a stale
   `VTune_PATH` reference (`/opt/intel/oneapi/vtune/2026.1/bin64/..`)
   that `set_and_check()` rejected because of the `..`. Mitigation:
   disable `GINKGO_HAVE_VTUNE=0` and comment out the `set_and_check`
   line in the installed config. (This is a CMake drift between Ginkgo
   build time and re-use; not a bug.)

2. `gko::share()` requires an rvalue in Ginkgo 2.0 — `std::move(unique_ptr)`
   needed before `gko::share()`. The Ginkgo 1.x examples that use bare
   `gko::share(unique_ptr&)` fail to compile in 2.0 (same kind of
   migration as we ran into for `Ilu<ir, ir>` template form in
   [Finding 24](24_pr168_patched_ilu_first_test.md)).

## Implications

### For the Pioneer story on this repo

The 1.10/1.11 era of three "SYCL algorithm bugs" (Findings 02 + 05 + 18 +
19) has effectively **ended with Ginkgo 2.0**. The chain of upstream
work that closed each gap:

- KIT clarification on Par* vs IC/ILU naming (May 2026, finding 05 corrected)
- nbeams' [PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
  for `lower_trs`/`upper_trs` via oneMKL trsm (merged 2026-06-02)
- Other 2.0 development fixes for `find_blocks` and `add_candidates`
  (not separately attributed here — confirmed by behaviour-test)

The remaining hardware-level limit (32 GB VRAM on BMG-G31 vs >32 GB
required for ILU/GMRES on 34M cells, see [Findings 22](22_vram_pressure_gmres_oom.md)
and [24](24_pr168_patched_ilu_first_test.md)) and the OGL multi-rank
gate ([Finding 25](25_cr_26.18_multirank_pthread_race.md)) are the
only blockers between us and a working strong-preconditioner GPU CFD
pipeline on Battlemage.

### For Ginkgo upstream (constructive)

Issue [#2015](https://github.com/ginkgo-project/ginkgo/issues/2015)
can be **closed** for the `find_blocks` and `add_candidates` items
based on this standalone confirmation. The ISAI / ICT `int32` overflow
beyond 4M rows is a separate, lower-priority item (workaround exists
via compile flag).

### For OGL upstream

With the algorithm side proven on Ginkgo 2.0, completing the OGL → 2.0
migration in [PR #168](https://github.com/hpsim/OGL/pull/168) (we
submitted two minimal patches as a comment to
[hpsim/OGL#170](https://github.com/hpsim/OGL/issues/170)) becomes the
highest-leverage upstream item for CFD-on-BMG.

## Files

- `findings/code/ginkgo_precond_sweep.cpp` — the test source (preserved
  for reproduction)
- `findings/code/CMakeLists.txt` — build setup
- [`logs/ginkgo-2.0-standalone/`](../logs/ginkgo-2.0-standalone/)
  contains the full sweep outputs for N=100, 500, 1000, 2000, 3000
