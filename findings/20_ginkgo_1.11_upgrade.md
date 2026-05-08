# Finding 20: Ginkgo 1.11 Upgrade Path on BMG-G31 via KIT Branch

## Status

Successful upgrade Ginkgo 1.10.0 → 1.11.0 on BMG-G31 with two patches
against oneAPI 2026.0.

For test results and bug-persistence outcome see
[Finding 19](19_ginkgo_111_upgrade_bug_persists.md). This document focuses
on the **build procedure** and what changed.

## Method

KIT-supported maintenance branch used instead of raw upstream tag:

```cmake
# /opt/ogl-src/CMakeLists.txt:77
set(GINKGO_CHECKOUT_VERSION
    "ogl_0600_gko110"     # was: ogl_0600_gko190 (= 1.10.0 + KIT patches)
    CACHE STRING "Use specific version of ginkgo")
```

This brings:

- Ginkgo 1.11.0 with bugfixes #1875 (MPICH-based MPI) and #1877 (ParILUT
  threshold select)
- KIT-applied OGL patches preserved (no API mismatch with the OGL wrapper
  layer)

The KIT naming convention `ogl_0600_gkoXXX` does not directly map to
Ginkgo version digits — `gko190` is the 1.10.0 maintenance branch,
`gko110` is the 1.11.0 maintenance branch. Confirm by reading
`CMakeLists.txt:project(... VERSION ...)` after fetch.

## Required Patches against icpx 2026.0

Two patches were necessary; both reflect upstream icpx changes that
KIT/Ginkgo upstream has not caught up with for this branch yet.

### Patch 1: deprecated `-fsycl-device-lib=all` flag

```diff
# /opt/ogl-src/build/_deps/ginkgo-src/dpcpp/CMakeLists.txt:149
-target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)
+# Removed for icpx 2026: -fsycl-device-lib=all
```

The flag was removed in icpx 2026; build of `libginkgo_dpcpp.so` link step
fails without this patch. icpx 2026 includes all device libs implicitly.

### Patch 2: `foam-shim/` include path

```diff
# /opt/ogl-src/include/CMakeLists.txt
 set_property(TARGET OGL_public_api APPEND PROPERTY
   INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")
+
+# foam-shim: Local shim for OpenFOAM Foundation 13 (fvCFD.H removed upstream)
+set_property(TARGET OGL_public_api APPEND PROPERTY
+  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/foam-shim")
```

`/opt/ogl-src/include/foam-shim/fvCFD.H` is an untracked local shim that
re-bundles OpenFOAM headers removed in OF Foundation 13. Without the
explicit include path, OGL-wrapper compilation fails with
`fatal error: 'fvCFD.H' file not found`.

## Verification

```
ldd libOGL.so | grep ginkgo
  libginkgo.so.1.11.0
  libginkgo_dpcpp.so.1.11.0    (553 MB, KIT-patched)
  libginkgo_omp.so.1.11.0
  libginkgo_reference.so.1.11.0
  libginkgo_device.so.1.11.0
  + cuda/hip dummies
```

OGL initialization at runtime reports:

```
Ginkgo version:  1.11.0 ( develop)
Ginkgo commit:   ogl_0600_gko110
MPI is GPU aware: 1     ← was 0 in 1.10
```

## BJ(1) Smoke Test — Performance Neutral

| Configuration | s/Step | Iterations |
|---|---|---|
| Ginkgo 1.10 BJ(1) (baseline) | 53.5 | 200 (cap) |
| Ginkgo 1.11 BJ(1)            | 53.2 | 201-202 (cap) |

Performance-neutral within measurement noise. API-compatibility validated.

## Pioneer Note

OGL initialization in 1.11 reports `MPI is GPU aware: 1` (was `0` in 1.10),
but `forceHostBuffer=true` is still required and the OGL log still shows
`Forces host buffer based communication: 1`. The change is in capability
detection, not in the runtime path the solver actually takes. No
performance impact observed for our case.

## Files

- [`logs/stufe4-ginkgo111/teil-a-1.11.log`](../logs/stufe4-ginkgo111/teil-a-1.11.log)
  — runtime log showing version detection
- [`logs/ginkgo-1.11-test/log.ginkgo111-bj1`](../logs/ginkgo-1.11-test/log.ginkgo111-bj1)
  — full BJ(1) smoke-test trace (3 timesteps)
- Build patches documented above; both carry over until upstream KIT or
  Ginkgo merges the icpx 2026 fix.
