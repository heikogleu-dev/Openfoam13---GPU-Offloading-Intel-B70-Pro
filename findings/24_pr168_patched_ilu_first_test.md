# Finding 24: PR #168 (Patched) Successfully Builds — First ILU Test on Ginkgo 2.0 + BMG-G31

## TL;DR

After two minimal local patches to PR #168 (`nbrPatchIndex` rename for
OF Foundation 13, ILU template-form migration to Ginkgo 2.0), **OGL
builds against Ginkgo 2.0 develop branch and BJ(1) runs cleanly on
BMG-G31**. This is the first working OGL/Ginkgo-2.0 stack on this
hardware generation.

ILU then reaches the preconditioner-generate phase — proving that the
`lower_trs` gap closed by [PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
is no longer the blocker — **but crashes with `DEVICE_LOST` during a
CSR→COO conversion inside ILU setup, at VRAM 26.5+ GB / 29 GB usable**.
This is the same hardware OOM pattern as
[Finding 22](22_vram_pressure_gmres_oom.md) for GMRES: 32 GB VRAM on
BMG-G31 is insufficient for ILU L+U storage on a 34M-cell automotive
mesh at np=8.

## Build — first successful OGL + Ginkgo 2.0 on BMG-G31

Built against PR #168 (`hpsim/OGL#168` HEAD `1000981`) + Ginkgo `develop`
fetched via CPM. **Six adjustments** required, two of them new code
patches:

### Code patches against PR #168 (new this round)

```diff
# src/MatrixWrapper/HostMatrix.cpp:239
-            const label neighbPatchId = patch.nbrPatchID();
+            const label neighbPatchId = patch.nbrPatchIndex();
```

OF Foundation 13 renamed `cyclicFvPatch::nbrPatchID()` to `nbrPatchIndex()`.
One-line fix.

```diff
# include/OGL/Preconditioner.hpp:270
-                gko::preconditioner::Ilu<ir, ir>::build()
+                gko::preconditioner::Ilu<scalar, false, label>::build()
```

Ginkgo 2.0 changed `Ilu` template signature from
`Ilu<L_solver, U_solver, ...>` (1.x type-parameters) to
`Ilu<ValueType, ReverseApply, IndexType>` (2.x). Runtime `with_l_solver`
/ `with_u_solver` builder calls still work — only the template head
changed. With `scalar=double`, `label=int32` (Foam types), this is
equivalent to `Ilu<>::build()` with defaults; we kept it explicit for
clarity.

### Environmental adjustments (same as [Finding 23](23_pr168_ginkgo_2.0_migration_test.md))

- `cmake --preset release` (not debug — debug burns 38 GB RSS in
  sycl-post-link)
- `-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON` (else `install(TARGETS ginkgo)`
  fails when a system Ginkgo is also present)
- Remove `-fsycl-device-lib=all` from fetched `dpcpp/CMakeLists.txt:149`
  (deprecated in icpx 2026)
- Add `foam-shim/` to `OGL_public_api` include directories (OF Foundation 13
  no longer ships `fvCFD.H`)

With those six adjustments, build completes:

```
[372/372] Linking CXX shared library libOGL.so
ldd libOGL.so | grep ginkgo:
  libginkgo.so.2.0.0          → installed
  libginkgo_dpcpp.so.2.0.0    → 248 MB, KIT-patched
  libginkgo_omp.so.2.0.0
  libginkgo_reference.so.2.0.0
  libginkgo_device.so.2.0.0
  + cuda/hip dummies
```

## Test 1 — BJ(1) smoke on Ginkgo 2.0 ✅

```
[OGL LOG][lduLduBase.hpp:106] Initialising OGL
    Ginkgo version: 2.0.0 ( develop)
    ...
BJsyclGKOCG:  Solving for p, Initial residual = 1, Final residual = 0.8738505,
              No Iterations 201
```

Single time-step (endTime=1 smoke), exit RC=0. Iteration count and
residual pattern identical to the Ginkgo 1.11 baseline ([Finding 19](19_ginkgo_111_upgrade_bug_persists.md)).
The OGL + Ginkgo-2.0 plumbing is functional.

## Test 2 — ILU on Ginkgo 2.0 + PR #2023's `lower_trs` ❌ VRAM OOM

```
[OGL LOG][Preconditioner.hpp:219] Generate preconditioner ILU
...
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

Stack (extracted):

```
sycl::_V1::exception thrown
  ↑ libsycl.so.9 (UR error 20)
  ↑ libginkgo_dpcpp.so.2.0.0  DpcppExecutor::raw_copy_to
  ↑ libginkgo.so.2.0.0        Csr<>::convert_to(Coo)
```

**VRAM behaviour:** Observed live via Ubuntu Resources tool during
the run:

1. Long *stable plateau* at ~26.5 GB / 32 GB raw (≈29 GB usable) during
   ILU setup with p + L + U on device
2. **Spike > 32 GB** during `Csr::convert_to(Coo)` (visible at top of
   stack trace) → SYCL `DEVICE_LOST`

The stack pinpoints the spike: format-conversion temporarily duplicates
a matrix in COO form alongside its CSR source.

### Theoretical vs measured VRAM

Per rank (np=8, 4.25M cells), with ~7 NNZ/row, double precision:

| Component | Size |
|---|---|
| p-matrix CSR (val + col_idx + row_ptr) | 475 MB/rank → ~3.8 GB total |
| L factor (ILU) | ~3.8 GB |
| U factor (ILU) | ~3.8 GB |
| CG Krylov vectors (~5) | 170 MB/rank → 1.4 GB |
| Workspace + overhead | ~2 GB |
| **Theoretical total** | **~15 GB** |

Theoretical 15 GB ≪ 32 GB available — yet we observe ~26.5 GB plateau
then >32 GB spike. **3× higher than expected.** Plausible causes:

1. **CSR → COO conversion duplication** (the immediate trigger from
   the stack trace). COO needs 3 arrays of length NNZ alongside the
   source CSR → ~3.8 GB extra per matrix conversion. Doing this near
   the 26.5 GB plateau pushes over the 32 GB limit.
2. **`ranksPerGPU=8` workspace duplication.** Each rank has its own
   DPC++ device context on the same GPU; Ginkgo's per-rank executor
   may allocate workspace independently rather than sharing.
3. **OGL distributed-matrix overhead** — same fixed-overhead pattern
   already documented in [Finding 22 (GMRES OOM)](22_vram_pressure_gmres_oom.md):
   GMRES showed ~26 GB fixed memory upfront, independent of `krylovDim`.
   Likely shared root cause.

**Most likely combined diagnosis:** baseline 15-20 GB (matrices +
overhead) plus ~5-10 GB transient spike from CSR→COO conversion = >32 GB
OOM. A non-duplicating in-place conversion path, or smaller-batched
conversion, would let ILU fit comfortably.

This is the same pattern type as [Finding 22 (GMRES OOM)](22_vram_pressure_gmres_oom.md):
SYCL backend cannot satisfy the spike allocation, surfaces the failure
as `DEVICE_LOST` rather than as a clean `gko::AllocationError`.

## What this tells us

| Question | Answer |
|---|---|
| Does PR #168 + 2 minimal patches build OGL against Ginkgo 2.0 on BMG-G31? | ✅ Yes |
| Does the new OGL/Ginkgo-2.0 stack run the BJ(1) baseline cleanly? | ✅ Yes, identical iter/residual to Ginkgo 1.11 |
| Does the `lower_trs` gap fixed by [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023) unblock the ILU code path? | ✅ Yes — ILU reaches "Generate preconditioner ILU" (was `NotImplemented` immediately in 1.x) |
| Does ILU now actually run a solve on 34M cells + np=8? | ❌ No — VRAM stable plateau ~26.5 GB, then >32 GB spike during CSR→COO conversion → OOM |
| Is the OOM a Ginkgo bug or a hardware constraint? | Mixed. Theoretical bedarf is ~15 GB; observed 3× higher. Hardware is structurally tight, but a non-duplicating conversion path inside Ginkgo would likely keep ILU under the 32 GB ceiling. Worth investigating upstream. |
| Diagnostic-quality issue with `DEVICE_LOST` vs `gko::AllocationError`? | Yes — same as Finding 22. Constructive feedback for upstream. |

## Cross-version comparison of `BJ(1)` baseline

| Build | s/Step | Iterations | Outcome |
|---|---|---|---|
| Ginkgo 1.10 ([Finding 02](02_bj_blocksize_int_underflow.md), baseline) | 53.5 | 200 (cap) | ✅ |
| Ginkgo 1.11 KIT branch ([Finding 19](19_ginkgo_111_upgrade_bug_persists.md)) | 53.2 | 201 (cap) | ✅ |
| Ginkgo 2.0 develop + PR #168 patched (this finding, smoke) | n/a (1-step) | 201 (cap) | ✅ |

Ginkgo-2.0 BJ(1) iteration count matches; no behavioural regression.
Performance comparison needs a multi-step run (deferred — current focus
was the build/smoke).

## System state

- Installed `libOGL.so`: now the **PR #168 patched + Ginkgo 2.0 build**
  (Jun 9 23:11, 3.09 MB)
- All Ginkgo 2.0.0 libs in `FOAM_USER_LIBBIN`:
  `libginkgo*.so.2.0.0` (incl. `libginkgo_dpcpp.so.2.0.0` 248 MB)
- Backups preserved for instant rollback:
  - `libOGL.so.1.11.bak.20260522-2242` (Ginkgo 1.11 baseline)
  - `libOGL.so.1.10.bak.20260507-2145` (Ginkgo 1.10 baseline)

## Implications for upstream

### For Ginkgo project (constructive)

- `lower_trs`/`upper_trs` via oneMKL trsm (PR #2023) is now reachable on
  BMG-G31 — please consider this confirmation that the merge unblocked
  one Battlemage-specific path
- `DEVICE_LOST` raised by the L0 backend during a Csr→Coo conversion is
  almost certainly an allocation failure; a guard in
  `DpcppExecutor::raw_copy_to` / `raw_alloc` that surfaces this as a
  clean `gko::AllocationError` would help users distinguish OOM from
  genuine driver-level device loss

### For OGL project ([hpsim/OGL#170](https://github.com/hpsim/OGL/issues/170))

PR #168 needs the two minimal patches above to build against the
current OF Foundation 13 / Ginkgo 2.0 develop stack on icpx 2026. We
are happy to PR both, especially the trivial `nbrPatchIndex` rename.

### For pioneers picking up Battlemage CFD

32 GB BMG-G31 is structurally undersized for strong-preconditioner CFD
on automotive-scale meshes (34M cells + np=8). The pattern is now
observed for both GMRES ([Finding 22](22_vram_pressure_gmres_oom.md))
and ILU (this finding). CG + weak preconditioners stay within bounds;
all the strong options (ILU, ParIc, Multigrid hierarchies) need more
than 32 GB on this mesh class.

## Files

- [`logs/pr168-patched/build_success.log`](../logs/pr168-patched/build_success.log) — 12 MB full build log of the successful release build
- [`logs/pr168-patched/bj1-smoke.log`](../logs/pr168-patched/bj1-smoke.log) — BJ(1) smoke run, Ginkgo 2.0
- [`logs/pr168-patched/ilu-smoke.log`](../logs/pr168-patched/ilu-smoke.log) — ILU first crash (initial run, no sampler)
- [`logs/pr168-patched/ilu-smoke-vram.log`](../logs/pr168-patched/ilu-smoke-vram.log) — ILU crash with VRAM sampler attempt (sampler died before save — VRAM peak observed live via Ubuntu Resources tool: 26.5+ GB)

## Status

PR #168 is **functionally close** to mergeable for the OF Foundation 13
/ Ginkgo 2.0 / Intel Battlemage stack — needs the two minor patches in
this finding. ILU on this hardware is **algorithmically possible now**
but **hardware-bound** for the 34M-cell case. Re-test on a smaller mesh
(≤8M cells) would isolate whether ILU itself works correctly given
sufficient VRAM headroom.
