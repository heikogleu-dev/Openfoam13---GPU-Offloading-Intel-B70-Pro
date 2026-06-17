# Finding 26: Ginkgo 2.0 Standalone SYCL Preconditioner Sweep — Three Bugs Are FIXED

## TL;DR

A standalone C++ test program against the installed Ginkgo 2.0 SYCL
backend on BMG-G31 (no OGL, no OpenFOAM, no MPI multi-rank — bypasses
the [CR 26.18 multi-rank gate](25_cr_26.18_multirank_pthread_race.md))
confirms that **three previously documented Ginkgo SYCL bugs are fixed
in Ginkgo 2.0**:

| Bug | 1.10 / 1.11 status | 2.0 status |
|---|---|---|
| `dpcpp::jacobi::find_blocks` `size_t` underflow ([Finding 02](02_bj_blocksize_int_underflow.md)) | crash, BJ(>1) unusable | ✅ **FIXED** — BJ(2), BJ(4), BJ(8), BJ(16) all run up to 36M rows |
| `dpcpp::par_ict_factorization::add_candidates` SIGABRT ([Finding 05](05_sycl_preconditioner_status.md)) | SIGABRT, ICT unusable | ✅ **FIXED** — ICT runs up to 4M rows (int32 overflow caps higher) |
| `dpcpp/solver/lower_trs_kernels.dp.cpp:43: generate is not implemented` ([Finding 05](05_sycl_preconditioner_status.md)) | NotImplemented, ILU/IC unusable | ✅ **FIXED** by [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023) — ILU runs up to 36M rows |
| Multigrid PGM coarsening OOM + divergence ([Finding 08](08_multigrid_device_lost.md)) | OOM, MG unusable | ✅ **FIXED** — Multigrid runs up to ~25M rows (above 32 GB ceiling at 36M) |

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

## VRAM scaling — how many rows fit in 32 GB?

A follow-up sweep added a `gko::log::Logger`-derived `AllocTracker` to
the test program that counts every Ginkgo device allocation. Per-(N,
precond) peak allocated bytes were recorded across N=100..6000
(10k..36M rows). Note: this measures **allocations Ginkgo itself made
through its executor**, which is a lower bound — it doesn't include
SYCL runtime overhead, kernel-launch scratch, or driver-side device
state. Real `vram_mm` usage is slightly higher.

### Per-row allocation rate (from N=2000 baseline, with proper alloc+free tracking)

**Update 2026-06-17:** Extended to BJ(4), BJ(8), BJ(16), Multigrid +
fixed the tracker (was monotonic-increasing; now properly decrements
on free via an `addr → bytes` map).

| Preconditioner | Bytes / row | Implied max rows in 27 GB usable | 34M-cell standalone footprint |
|---|---|---|---|
| BJ(maxBlockSize=1) | 48 | ~600M rows | 1.6 GB |
| BJ(maxBlockSize=2) | 84 | ~343M rows | 2.9 GB |
| **BJ(maxBlockSize=4)** | **100** | **~290M rows** | **3.4 GB** |
| **BJ(maxBlockSize=8)** | **132** | **~218M rows** | **4.5 GB** |
| **BJ(maxBlockSize=16)** | **196** | **~147M rows** | **6.7 GB** |
| ILU (ParIlu + lower_trs) | 268 | ~108M rows | 9.2 GB |
| ICT (ParIct + Ic apply) | 832 | ~35M rows | 28 GB (`int32` overflow ahead at ~9M) |
| ISAI sparsityPower=1 | 136 | ~212M rows | 4.7 GB |
| **Multigrid (PGM + BJ smoother)** | **1027** | **~28M rows** | **35 GB** (above 32 GB ceiling for 34M cells) |

### Measured at each N

| N | Rows | BJ(1) GB | BJ(2) GB | ILU GB | ICT GB | ISAI GB |
|---|---|---|---|---|---|---|
| 100 | 10k | 0.001 | 0.001 | 0.003 | 0.026 | 0.002 |
| 500 | 250k | 0.013 | 0.029 | 0.075 | 0.663 | 0.041 |
| 1000 | 1M | 0.052 | 0.116 | 0.302 | 2.649 | 0.164 |
| 2000 | 4M | 0.209 | 0.466 | 1.207 | 10.635 | 0.656 |
| 3000 | 9M | 0.469 | 1.048 | 2.716 | 16.293 (int32-ovf) | 1.476 |
| 4000 | 16M | 0.834 | 1.863 | 4.828 | 28.939 (int32-ovf) | 2.623 |
| 5000 | 25M | 1.304 | 2.910 | 7.544 | 29.876 (int32-ovf) | 4.098 |
| 6000 | 36M | 1.878 | 4.191 | 10.863 | 12.609 (int32-ovf) | 5.901 |

ICT memory grows so fast it nearly exhausts 32 GB by N=4000 (16M rows)
even before the `int32` overflow halts setup. BJ(1) and ISAI scale
favourably; ILU is the inflection point for practical CFD pressure
systems.

### Cross-check with OGL multi-rank (np=8) data

Compared with the OGL distributed-matrix overhead documented in
[Finding 22 (GMRES OOM)](22_vram_pressure_gmres_oom.md) and
[Finding 24 (PR #168 ILU OOM)](24_pr168_patched_ilu_first_test.md):

- BJ(1) standalone @ 34M = 1.9 GB · OGL+np=8 @ 34M = ~9 GB
  ([Finding 02](02_bj_blocksize_int_underflow.md)) — **~4.7× distributed overhead**
- ILU standalone @ 34M = 11 GB · OGL+np=8 @ 34M = >32 GB spike
  ([Finding 24](24_pr168_patched_ilu_first_test.md)) — **>3× overhead** plus
  the Csr→Coo conversion spike at setup

So the distributed/OGL pipeline costs roughly 3-5× over what Ginkgo
allocates by itself. Useful as a rule of thumb when scaling future
cases.

### Practical headroom on B70 Pro (32 GB raw / 27 GB usable for SYCL)

For **single-rank standalone** Ginkgo on BMG-G31:

- BJ(1) / BJ(2) / ISAI: can comfortably handle 100M+ row problems
- ILU: comfortable up to ~80-90M rows (well above our 34M test case)
- ICT: limited to ~10M rows by sheer memory growth, plus the `int32`
  index overflow blocks anything past ~9M

For **OGL multi-rank** at np=8 on the same 32 GB GPU, derate by ~3-5×
distributed-matrix overhead — so the practical ceiling for ILU drops to
roughly 15-25M cells without a smaller-mesh strategy. Our 34M-cell test
case sits exactly at that boundary, which is why ILU OOMs in OGL+np=8
but runs cleanly standalone.

## Files

- `findings/code/ginkgo_precond_sweep.cpp` — the test source (with the
  AllocTracker logger; preserved for reproduction)
- `findings/code/CMakeLists.txt` — build setup
- [`logs/ginkgo-2.0-standalone/`](../logs/ginkgo-2.0-standalone/)
  contains the full sweep outputs for N=100, 500, 1000, 2000, 3000
- [`logs/ginkgo-2.0-standalone/vram-sweep.csv`](../logs/ginkgo-2.0-standalone/vram-sweep.csv)
  contains the (N, preconditioner, status, time, peak-bytes) table
