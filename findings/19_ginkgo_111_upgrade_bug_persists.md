# Finding 19: Ginkgo 1.11.0 Upgrade — `find_blocks` Underflow Persists

## Hypothesis Tested

Upgrade Ginkgo 1.10 → 1.11 to verify whether bugfixes #1875 (MPICH-based MPI)
or #1877 (ParILUT threshold select) — or any side-effect on adjacent
Jacobi block-counting code paths — fix the deterministic `size_t` underflow
in `dpcpp::jacobi::find_blocks` for `BJ(maxBlockSize > 1)` documented in
[finding 02](02_bj_blocksize_int_underflow.md).

## Test Setup

- Hardware: Intel Arc Pro B70 Pro (BMG-G31, Device `0xE223`)
- Stack: CR 26.05.37020.3-1, IGC 2.32.7, oneAPI 2026.0
- OGL: dev branch (commit `15f81d2+`)
- Mesh: 34M cells, np=8

## Build

OGL-CMake `GINKGO_CHECKOUT_VERSION` switched from `ogl_0600_gko190` (Ginkgo
1.10.0 + KIT patches) to `ogl_0600_gko110` (Ginkgo 1.11.0 + KIT patches).
Both branches are KIT-supported OGL forks; no raw upstream `v1.11.0` was
used to preserve the OGL-specific `dont expose add_operator` patch.

Build required two patches against icpx 2026.0:

1. Removed `-fsycl-device-lib=all` from `dpcpp/CMakeLists.txt:149`
   — the flag was removed in icpx 2026 (now implicit default).
2. Added `${CMAKE_CURRENT_SOURCE_DIR}/foam-shim` to `OGL_public_api`
   include directories — the local `foam-shim/fvCFD.H` (re-bundles headers
   removed in OF Foundation 13) was untracked and needed an explicit
   include path.

`libginkgo_dpcpp.so.1.11.0` (553 MB) and `libOGL.so` linked successfully;
`ldd libOGL.so | grep ginkgo` confirms all 7 Ginkgo libs resolve to
`*.1.11.0`.

## Result A: BJ(1) Smoke-Test — API Compatible, Performance Neutral

| Metric | Ginkgo 1.10 | Ginkgo 1.11 |
|---|---|---|
| s/Step (mean of step 2-3) | 53.5 | **53.2** |
| Iterations / step | 200 (cap) | 201-202 (cap) |
| Run completes | ✅ | ✅ |
| `MPI is GPU aware` (OGL log) | 0 | **1** ⚠️ |

OGL header reports `MPI is GPU aware: 1` in 1.11 vs `0` in 1.10.
`forceHostBuffer=true` is still set in `fvSolution` and the OGL log still
shows `Forces host buffer based communication: 1` — so the change is in
how OGL/Ginkgo report MPI capability detection, not in the runtime path
the solver actually takes. No performance impact observed.

## Result B: BJ(maxBlockSize=2) — Bit-Identical Underflow

```
terminate called after throwing an instance of 'gko::AllocationError'
  what(): /opt/ogl-src/build/_deps/ginkgo-src/dpcpp/base/executor.dp.cpp:104:
          DPC++: failed to allocate memory block of 18446744073709551615B
```

Stack frame (from `logs/ginkgo-1.11-test/log.ginkgo111-bj2`):

```
libginkgo_dpcpp.so.1.11.0
  → gko::kernels::dpcpp::jacobi::find_blocks<double, int>
    → DpcppExecutor::raw_alloc(18446744073709551615)
```

Identical mangled symbol, identical underflow value, identical crash phase
(preconditioner generate, before first solve iteration), identical exit
code (RC=134, SIGABRT). The only difference vs the 1.10 trace is the lib
path (`...so.1.11.0` instead of `...so.1.10.0`).

## Conclusion

The `size_t` underflow in `dpcpp::jacobi::find_blocks` is **deterministic
across Ginkgo 1.10 and 1.11**, and **across Level Zero V1 and V2 adapters**
(see [finding 18](18_v2_adapter_ruled_out.md)).

The bug survives both:
- A full Ginkgo minor-version upgrade
- The Unified Runtime adapter generation switch

It must therefore live in the SYCL/DPC++ implementation of
`preconditioner/jacobi/find_blocks` (or a kernel it calls), not in the
runtime layers below or in adjacent generations of the surrounding code.

## Combined Pioneer Evidence Sheet

After this round, the find_blocks underflow is reproducible-confirmed on:

| Variant | Ginkgo | L0 Adapter | Outcome |
|---|---|---|---|
| Default | 1.10 | V2 (default) | underflow, 8.41 GB peak |
| V1 forced | 1.10 | V1 (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) | underflow, 8.47 GB peak |
| Upgrade | 1.11 | V2 (default) | **underflow, identical signature** |

## Files

- [`logs/ginkgo-1.11-test/log.ginkgo111-bj1`](../logs/ginkgo-1.11-test/log.ginkgo111-bj1)
  — full BJ(1) baseline run (3 timesteps, ~53 s/step)
- [`logs/ginkgo-1.11-test/log.ginkgo111-bj2`](../logs/ginkgo-1.11-test/log.ginkgo111-bj2)
  — full BJ(2) crash trace with stack frame

## Build Patches (carried locally for reproduction)

```diff
# /opt/ogl-src/CMakeLists.txt
-set(GINKGO_CHECKOUT_VERSION "ogl_0600_gko190" ...)
+set(GINKGO_CHECKOUT_VERSION "ogl_0600_gko110" ...)

# /opt/ogl-src/build/_deps/ginkgo-src/dpcpp/CMakeLists.txt:149
-target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)
+# Removed for icpx 2026: -fsycl-device-lib=all (now implicit default)

# /opt/ogl-src/include/CMakeLists.txt — added after binary-dir include:
+set_property(TARGET OGL_public_api APPEND PROPERTY
+  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/foam-shim")
```

## Status

Datapoint contributed to `ginkgo-project/ginkgo` `find_blocks` underflow
report (extending evidence in finding 02 + 18). The bug is now
characterised as a deterministic `dpcpp/preconditioner/jacobi` issue
independent of Ginkgo minor version and L0 adapter generation.
