# PR #168 Test Results on Intel Arc Pro B70 Pro

Following up on your suggestion in [Issue #170](https://github.com/hpsim/OGL/issues/170)
to test PR #168 against the Ginkgo `develop` branch. Build did not complete —
two compile errors block before any preconditioner test can be run.

## System

- GPU: Intel Arc Pro B70 Pro (BMG-G31, Battlemage Xe2, 32 GB GDDR6)
- OS: Ubuntu 26.04 LTS, Kernel 7.0.0-15
- oneAPI: 2026.0.0, icpx (IntelLLVM@2026.0.0), IGC 2.32.7
- Compute Runtime: 26.05.37020.3-1 (pinned — 26.14 has multi-rank issues)
- OGL: PR #168 (HEAD `1000981` — "add license", + 4 commits over `dev`)
- Ginkgo: `develop` branch via CPM → fetched as **2.0.0**
- OpenFOAM: Foundation 13

## Build Procedure Used

Followed your suggestion modulo two non-default adjustments:

```bash
CC=icx CXX=icpx cmake --preset release \
    -DOGL_GINKGO_CHECKOUT_VERSION=develop \
    -DOGL_BUILD_SYCL=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON
cmake --build --preset release -j $(nproc)
```

Notes:

- `release` instead of `debug` preset: with `--preset debug`, the icpx 2026
  `sycl-post-link` step ran for 75+ min with RSS climbing past 38 GB on a
  single instance before we aborted — known icpx behaviour with `-O0` and
  ~120 SYCL kernels. `release` (`-O3`) is well-behaved (RSS ~0.6 GB).
- `-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON`: with a system-installed Ginkgo
  present, `find_package(Ginkgo QUIET)` in `cmake/CxxThirdParty.cmake`
  succeeds and skips the CPM fetch, but the `install(TARGETS ginkgo ...)`
  statement in `CMakeLists.txt:246` then errors with
  `install TARGETS given target "ginkgo" which does not exist`. Forcing the
  CPM path with this CMake-builtin makes the configure succeed.

Additionally, two local patches against icpx 2026 / OpenFOAM Foundation 13:

1. Removed `target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)`
   from the fetched `dpcpp/CMakeLists.txt:149`. The flag was removed in
   icpx 2026 (now implicit default); link of `libginkgo_dpcpp.so.2.0.0`
   fails without this patch.
2. Added `${CMAKE_CURRENT_SOURCE_DIR}/foam-shim` to `OGL_public_api`
   include directories. Local `include/foam-shim/fvCFD.H` re-bundles
   headers removed in OF Foundation 13; without this include path, OGL
   compilation fails with `fatal error: 'fvCFD.H' file not found`.

With those four adjustments, Ginkgo 2.0.0 builds cleanly (all 7 libs:
`libginkgo[_omp|_reference|_dpcpp|_cuda|_hip|_device].so.2.0.0`).
OGL wrapper compilation then hits two distinct errors:

## Error 1 — OpenFOAM Foundation 13 API drift

```
/opt/ogl-src/src/MatrixWrapper/HostMatrix.cpp:239:47:
    error: no member named 'nbrPatchID' in 'Foam::cyclicFvPatch'
  239 |             const label neighbPatchId = patch.nbrPatchID();
      |                                         ~~~~~ ^
```

Context (`HostMatrix.cpp:236-241`):

```cpp
#ifdef WITH_ESI_VERSION
            const label neighbPatchId = patch.neighbPatchID();
#else
            const label neighbPatchId = patch.nbrPatchID();
#endif
```

In OpenFOAM Foundation 13 (`finiteVolume/lnInclude/cyclicFvPatch.H:97`):

```cpp
virtual label nbrPatchIndex() const
{
    return cyclicPolyPatch_.nbrPatchIndex();
}
```

The method was renamed `nbrPatchID()` → `nbrPatchIndex()` in OF Foundation
between the OGL-supported version and v13. One-line fix:

```diff
-            const label neighbPatchId = patch.nbrPatchID();
+            const label neighbPatchId = patch.nbrPatchIndex();
```

## Error 2 — Ginkgo 2.0 API change in ILU preconditioner

```
/opt/ogl-src/include/OGL/Preconditioner.hpp:270:46:
    error: template argument for non-type template parameter must be an expression
  270 |                 gko::preconditioner::Ilu<ir, ir>::build()
      |                                              ^~
/opt/ogl-src/build/release/_deps/ginkgo-src/include/ginkgo/core/preconditioner/ilu.hpp:79:56:
    note: template parameter is declared here
```

This is the same kind of API break we documented for the 1.10 → 2.0 jump
in [our finding 10](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/10_ginkgo2_api_breaks.md):
in Ginkgo 2.0, `gko::preconditioner::Ilu` template parameters changed (the
`L_solver_type, U_solver_type` slots became a non-type parameter expecting
a value, not a type). The OGL code at `Preconditioner.hpp:270`:

```cpp
auto precond_factory =
    gko::preconditioner::Ilu<ir, ir>::build()
        .with_l_solver(gko::clone(trisolve_factory))
        .with_u_solver(gko::clone(trisolve_factory))
        .on(device_exec);
```

still uses the 1.x form. The error reproduces in 9 translation units that
include `Preconditioner.hpp`, blocking the OGL link entirely.

This is the gap PR #168 is meant to close — the Ginkgo 2.0 ILU migration
appears not yet complete in this branch.

## Preconditioner tests (TEIL C) — not run

Without a working `libOGL.so` on PR #168, the four planned smoke tests
(BJ maxBlockSize=16, ICT, Multigrid, hybrid matrix format distributed)
could not be executed. Our existing `libOGL.so` from the Ginkgo 1.11
build is untouched (install never ran), so the working baseline is
preserved.

## Summary for triage

| Item | Status |
|---|---|
| `cmake --preset release` configure | ✅ after `CMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON` |
| Ginkgo 2.0.0 build via CPM | ✅ after `-fsycl-device-lib=all` removal |
| OGL wrapper compile (foam-shim) | ✅ after local include path patch |
| `HostMatrix.cpp:239` (`nbrPatchID`) | ❌ OF Foundation 13 API rename — 1-line fix |
| `Preconditioner.hpp:270` (`Ilu<ir, ir>`) | ❌ Ginkgo 2.0 ILU migration incomplete |
| Preconditioner tests | ⏭ blocked |

Happy to retest as soon as the ILU migration lands — also happy to PR the
`nbrPatchIndex()` rename and `foam-shim` include path upstream if useful.

Build environment versions, all logs and the exact patches applied are
available on request; let me know what's most useful for narrowing this
down.

— Heiko
