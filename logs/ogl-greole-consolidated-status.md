# Consolidated status for hpsim/OGL#170 — PR #168 on Intel Arc Pro B70 (Battlemage) + OF Foundation 13 + Ginkgo 2.0

Hi @greole — consolidating several sessions of testing PR #168 on
Battlemage (BMG-G31) into one update. TL;DR: **PR #168 builds and runs
on this stack with two one-line patches; the Ginkgo-2.0 side is in good
shape; the remaining wall is an Intel Compute Runtime multi-process
bug below OGL, not OGL itself.**

Full public writeup (findings 23–29):
https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro

## 1. PR #168 builds + runs with two minimal patches

Stack: OF Foundation 13, Ginkgo `develop` (2.0.0) via CPM, oneAPI 2026.0
(icpx), CR pinned to 26.05 (see §4), BMG-G31.

Two source patches needed:

```diff
# src/MatrixWrapper/HostMatrix.cpp:239  (OF Foundation 13 API rename)
-            const label neighbPatchId = patch.nbrPatchID();
+            const label neighbPatchId = patch.nbrPatchIndex();
```

```diff
# include/OGL/Preconditioner.hpp:270  (Ginkgo 2.0 Ilu template form)
-                gko::preconditioner::Ilu<ir, ir>::build()
+                gko::preconditioner::Ilu<scalar, false, label>::build()
                     .with_l_solver(gko::clone(trisolve_factory))
                     .with_u_solver(gko::clone(trisolve_factory))
                     .on(device_exec);
```

With those, `libOGL.so` links against `libginkgo*.so.2.0.0` and runs:
**BJ(maxBlockSize=1) multi-rank (np=8) on a 34M-cell case completed a
timestep** (≈94 s/step, 200-iter cap, on a freshly-booted GPU).

Happy to open these as a PR against #168 if useful — both are trivial.

## 2. Four build-environment adjustments (not OGL bugs, but worth knowing)

For anyone reproducing on the current Intel stack:

1. **`cmake --preset release`, not `debug`** — with `-O0` the icpx 2026
   `sycl-post-link` step on Ginkgo's ~120 SYCL kernels grew past 38 GB
   RSS and effectively hung; `release` (`-O3`) finishes in ~10 min,
   ~0.6 GB.
2. **`-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON`** — with a system Ginkgo
   present, `find_package(Ginkgo QUIET)` in `cmake/CxxThirdParty.cmake`
   succeeds and skips CPM, but then `install(TARGETS ginkgo ...)` in
   `CMakeLists.txt` fails (`target "ginkgo" does not exist`). Might be
   worth guarding that install when Ginkgo was imported.
3. **Remove `-fsycl-device-lib=all`** from the fetched
   `dpcpp/CMakeLists.txt` — the flag was removed in icpx 2026; link
   fails otherwise. (Upstream Ginkgo issue, not OGL — flagging for
   awareness since CPM pins the Ginkgo branch.)
4. **`foam-shim/` include path** — OGL ships `include/foam-shim/fvCFD.H`
   for OF Foundation 13 (which removed `fvCFD.H`), but the include path
   isn't propagated to `OGL_public_api`; a 3-line `set_property` fixes
   it.

## 3. The `find_blocks` underflow is in OGL's distributed path, not the Ginkgo kernel

This is the most OGL-actionable finding. The `BJ(maxBlockSize>1)`
`size_t` underflow (`failed to allocate memory block of
18446744073709551615B` in `dpcpp::jacobi::find_blocks`) that we and
others have seen:

| Path | Result (Ginkgo 2.0) |
|---|---|
| Standalone `gko::matrix::Csr` + `Jacobi(maxBlockSize=2)`, single process, up to **36M rows** | ✅ runs clean |
| OGL distributed Schwarz wrapper, np=8, ~4.25M-row per-rank shard | ❌ identical underflow |

Same Ginkgo build, same kernel. A ~120-line standalone reproducer runs
BJ(2/4/8/16) fine to 36M rows
([findings/26](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/26_ginkgo_2.0_standalone_sweep.md)),
so the trigger is specifically how the distributed
`experimental::distributed::preconditioner::Schwarz` path feeds the
per-rank shard into `find_blocks` — possibly a shard with empty row
blocks or a sparsity pattern the standalone Poisson matrix doesn't
produce. Happy to dump the actual per-rank `Csr` shard (MatrixMarket)
if that helps reproduce it bit-identically.

## 4. The real wall: Intel CR 26.14–26.18 multi-process `zeInit` abort

Multi-rank OGL on BMG-G31 is gated by an Intel Compute Runtime
regression *below* OGL: CR ≥ 26.14 calls `abort()` in
`gmm_helper/resource_info.cpp:15` during `zeInit` whenever ≥2 processes
share the GPU (the normal MPI mode). Pure-Level-Zero minimal reproducer
(no SYCL/Ginkgo/OGL) + analysis:
[findings/29](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/29_cr_26.18_root_cause_pure_l0_multiprocess_abort.md),
upstream [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922).

Relevant for OGL because:
- It's why multi-rank needs CR 26.05 (we use a user-side
  `LD_LIBRARY_PATH` switch to an extracted 26.05 `libze_intel_gpu`, no
  sudo — [findings/27](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/27_cr2605_ld_switch_workaround.md)).
- We tried an `MPI_Barrier(MPI_COMM_WORLD)` in `ExecutorHandler::init()`
  before `DpcppExecutor::create` — **insufficient**, because the abort
  is in the CR driver during `zeInit`, reached before/independent of
  OGL's barrier. So there's no OGL-level fix for it; just flagging so
  you can point other BMG users at the CR pin.

## 5. What's verified vs pending

| Item | Status |
|---|---|
| PR #168 + 2 patches builds on OF13/Ginkgo2.0/icpx2026 | ✅ |
| BJ(1) multi-rank 34M cells runs (fresh GPU + CR 26.05) | ✅ **~51.5 s/step** (corrected, see update) |
| ILU reaches generate phase (PR #2023 `lower_trs` works) | ✅ generate reached |
| Strong-preconditioner (BJ8/ILU/MG) iteration counts + perf vs CPU GAMG | ✅ **measured — see update below** |
| `find_blocks` distributed-path underflow | ❌ open, localized to Schwarz path |

## Update (2026-06-17) — clean-boot multi-rank numbers

A fresh reboot + the CR 26.05 LD-switch produced deterministic 34M-cell
numbers (full writeup:
[findings/30](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/30_post_recovery_clean_multirank_perf.md)):

- **BJ(1): ~51.5 s/step** steady-state (the earlier "~94 s" was the
  setup-inclusive first step). Every pressure solve hits the 201-iter cap
  — never converges → ~1.44× slower than CPU GAMG (35.7 s/step).
- **ILU: VRAM OOM → `UR_RESULT_ERROR_DEVICE_LOST`** on the first solve,
  inside `ParIlu::generate_l_u` at `Csr::convert_to(Coo)`. Peak VRAM 31.5
  GB (fdinfo, summed over 8 ranks) — pinned to the 32 GB ceiling. ILU does
  not fit at 34M on a 32 GB card with the current OGL distributed
  overhead. The Csr→Coo materialization in the factorization is the spike;
  a leaner generate path (or a bigger card) would unblock it.
- **BJ(2–16): the `find_blocks` underflow reproduces** on Ginkgo 2.0 +
  CR 26.05 (clean `gko::AllocationError`, 16-EB request — GPU unharmed).

So the §5 "pending" row is answered: no strong preconditioner currently
works at 34M/32 GB — BJ1 too weak, ILU OOMs, BJ>1 underflows. The highest
-value OGL fix is still the distributed `find_blocks` path; second is a
VRAM-leaner ILU `generate`.

The Ginkgo `lower_trs` gap is closed
([PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023), merged
2026-06-02), so PR #168's Ginkgo-2.0 migration now has a real payoff —
working IC/ILU apply on SYCL. That makes finishing #168 genuinely
worthwhile for Intel-GPU users.

Glad to test patches, dump shards, or run debug builds — whatever helps
most. Thanks for the pointer to #168.

— Heiko
