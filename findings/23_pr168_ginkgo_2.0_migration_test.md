# Finding 23: PR #168 (OGL → Ginkgo 2.0 Migration) Build Test on BMG-G31

## Context

In response to greole's pointer in [hpsim/OGL#170](https://github.com/hpsim/OGL/issues/170)
suggesting the in-progress [PR #168](https://github.com/hpsim/OGL/pull/168)
as the Ginkgo 2.0 migration path, we tested whether PR #168 builds and
runs on the Battlemage G31 / OpenFOAM Foundation 13 / icpx 2026 stack.

**Outcome:** PR #168 does not currently build on this stack. Two distinct
compile errors block before any preconditioner test from the original
plan could run. The system remains on the working Ginkgo 1.11 install
(PR #168 build never reached `install`), so production is unaffected.

This finding documents the failure modes for greole and serves as a
checkpoint for the next pioneer who picks up PR #168 from a different
environment.

## What worked

| Stage | Status |
|---|---|
| `git fetch origin pull/168/head:pr168` + checkout | ✅ |
| `cmake --preset release -DOGL_GINKGO_CHECKOUT_VERSION=develop` | ✅ after 4 adjustments (see below) |
| Ginkgo 2.0.0 (develop) build via CPM | ✅ all 7 `libginkgo*.so.2.0.0` produced |
| OGL wrapper compile | ❌ two errors, see below |

`libginkgo_dpcpp.so.2.0.0` (Release-built) is 95 MB and links cleanly —
the Ginkgo side of the migration is sound. The break is in the OGL
wrapper code.

## Four build adjustments needed against current stack

These are not "PR #168 bugs" — they are environmental drift between the
PR-as-written and our target stack. Documented for reproducibility.

### Adjustment 1 — `release` preset instead of `debug`

The maintainer suggestion in #170 was `cmake --preset debug`. With debug
preset, the icpx 2026 `sycl-post-link` step ran for 75+ minutes with RSS
climbing past 38 GB on a single instance before we aborted — known icpx
behaviour with `-O0` and ~120 SYCL kernels. The `release` preset (`-O3`)
finishes the same step in ~10 minutes with RSS ~0.6 GB.

The `debug` preset is only useful for gdb-stepping; for run-tests, `release`
is the practical choice.

### Adjustment 2 — `-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON`

With a system-installed Ginkgo present, `find_package(Ginkgo QUIET)` in
`cmake/CxxThirdParty.cmake` succeeds and skips the CPM fetch. But the
`install(TARGETS ginkgo ...)` statement in `CMakeLists.txt:246` then
errors:

```
CMake Error at CMakeLists.txt:246 (install):
  install TARGETS given target "ginkgo" which does not exist.
```

Forcing the CPM path with the CMake-builtin `CMAKE_DISABLE_FIND_PACKAGE_Ginkgo`
makes the configure succeed. **Suggestion for PR #168:** either skip the
`install(TARGETS ginkgo ...)` when Ginkgo came from `find_package` (was
imported), or document the requirement explicitly.

### Adjustment 3 — Remove `-fsycl-device-lib=all`

```diff
# build/release/_deps/ginkgo-src/dpcpp/CMakeLists.txt:149
-target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)
+# Removed for icpx 2026: -fsycl-device-lib=all (now implicit default)
```

The flag was removed in icpx 2026 — link of `libginkgo_dpcpp.so.2.0.0`
fails without this patch. Same patch we needed for Ginkgo 1.11 — see
[Finding 20](20_ginkgo_1.11_upgrade.md).

### Adjustment 4 — `foam-shim/` include path

`/opt/ogl-src/include/foam-shim/fvCFD.H` is an untracked local shim that
re-bundles headers removed in OF Foundation 13. The PR #168 branch does
not include the shim in the `OGL_public_api` include path — same gap as
in `dev` branch, see [Finding 20](20_ginkgo_1.11_upgrade.md). Local patch:

```diff
# include/CMakeLists.txt — added after binary-dir include:
+set_property(TARGET OGL_public_api APPEND PROPERTY
+  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/foam-shim")
```

## Blocking error 1 — OpenFOAM Foundation 13 API rename

```
/opt/ogl-src/src/MatrixWrapper/HostMatrix.cpp:239:47:
    error: no member named 'nbrPatchID' in 'Foam::cyclicFvPatch'
  239 |             const label neighbPatchId = patch.nbrPatchID();
      |                                         ~~~~~ ^
```

OGL source `HostMatrix.cpp:236-241`:

```cpp
#ifdef WITH_ESI_VERSION
            const label neighbPatchId = patch.neighbPatchID();
#else
            const label neighbPatchId = patch.nbrPatchID();
#endif
```

OF Foundation 13 (`finiteVolume/lnInclude/cyclicFvPatch.H:97`) — the
method was renamed:

```cpp
virtual label nbrPatchIndex() const
{
    return cyclicPolyPatch_.nbrPatchIndex();
}
```

**One-line fix:** `nbrPatchID()` → `nbrPatchIndex()` in the
`#else` branch.

## Blocking error 2 — Ginkgo 2.0 ILU template-parameter change

```
/opt/ogl-src/include/OGL/Preconditioner.hpp:270:46:
    error: template argument for non-type template parameter must be an expression
  270 |                 gko::preconditioner::Ilu<ir, ir>::build()
      |                                              ^~
/opt/ogl-src/build/release/_deps/ginkgo-src/include/ginkgo/core/preconditioner/ilu.hpp:79:56:
    note: template parameter is declared here
```

OGL source `Preconditioner.hpp:269-273`:

```cpp
auto precond_factory =
    gko::preconditioner::Ilu<ir, ir>::build()
        .with_l_solver(gko::clone(trisolve_factory))
        .with_u_solver(gko::clone(trisolve_factory))
        .on(device_exec);
```

In Ginkgo 2.0, `gko::preconditioner::Ilu` template parameters changed —
`L_solver_type, U_solver_type` slots became a non-type parameter
expecting a value, not a type. The OGL code still uses the 1.x form. The
error propagates through 9 translation units that include
`Preconditioner.hpp`, blocking the OGL link entirely.

This is the gap PR #168 is meant to close — the Ginkgo 2.0 ILU migration
appears incomplete in this branch. Same kind of API break as
[Finding 10](10_ginkgo2_api_breaks.md) documented for the 1.10 → 2.0
jump.

## Preconditioner tests not run

The four tests planned for the PR #168 build (BJ maxBlockSize=16, ICT,
Multigrid, hybrid matrix format distributed — re-running the failures
from Findings 02/05/08/16 against the migrated stack) require a working
`libOGL.so`. Skipped.

## System state after test

- Installed `libOGL.so`: unchanged Ginkgo 1.11 build from
  [Finding 20](20_ginkgo_1.11_upgrade.md), `2026-05-07 22:30`
- `ldd libOGL.so | grep ginkgo`: confirms `libginkgo*.so.1.11.0` resolves
- Backups present: `libOGL.so.1.10.bak.20260507-2145`,
  `libOGL.so.1.11.bak.20260522-2242`
- OGL src tree on `pr168` branch with local patches (foam-shim include,
  `-fsycl-device-lib=all` removal); `dev`-branch state preserved in
  `git stash@{0}`

## Files

- [`logs/pr168-test/configure_release_success.log`](../logs/pr168-test/configure_release_success.log)
  — the only configure pass (after all four adjustments)
- [`logs/pr168-test/build_attempt3_foam_shim_patched.log`](../logs/pr168-test/build_attempt3_foam_shim_patched.log)
  — last build attempt with both blocking errors
- [`logs/pr168-test/build_attempt1_first_fail.log`](../logs/pr168-test/build_attempt1_first_fail.log),
  [`build_attempt2_release.log`](../logs/pr168-test/build_attempt2_release.log)
  — earlier attempts (12 MB each, kept for full Pioneer reproducibility)
- [`logs/pr168-test/issue_170_comment_ready_to_post.md`](../logs/pr168-test/issue_170_comment_ready_to_post.md)
  — issue body prepared for posting to hpsim/OGL#170

## Status

Issue-comment text ready for posting to [hpsim/OGL#170](https://github.com/hpsim/OGL/issues/170).
Will re-test as soon as the ILU 2.0 migration lands; happy to PR the
`nbrPatchIndex()` rename and `foam-shim` include path upstream
separately if useful.
