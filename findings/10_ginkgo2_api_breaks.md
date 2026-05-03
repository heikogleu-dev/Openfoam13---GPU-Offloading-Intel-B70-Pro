# Ginkgo 2.0 API Incompatibility with OGL

## Summary

OGL dev branch is incompatible with Ginkgo 2.0. Build fails with 36+ errors
at 3 specific breaking changes in Preconditioner.hpp.

## Breaking Changes

| Location | API Break | Impact |
|---|---|---|
| Preconditioner.hpp ~270 | `Ilu<ir,ir>` template signature changed | ICT precond setup |
| Preconditioner.hpp ~612 | `Schwarz::get_local_solver()` removed | Distributed caching |
| Preconditioner.hpp ~614,620 | `gko::UpdateMatrixValue` interface removed | Precond update path |

## Build Reproduction

```bash
mkdir -p /opt/ogl-ginkgo2/build && cd /opt/ogl-ginkgo2/build
cmake /opt/ogl-src \
  -DCMAKE_CXX_COMPILER=icpx \
  -DOGL_USE_EXTERNAL_GINKGO=ON \
  -DGinkgo_DIR=/opt/ginkgo/lib/cmake/Ginkgo
make -j$(nproc)
# → 36 errors, all in Preconditioner.hpp
```

Two CMake patches were needed first:
1. Bump version requirement: `ginkgo_find_package(Ginkgo "Ginkgo::ginkgo" FALSE 2.0.0)` (was 1.8.0; Ginkgo's `SameMajorVersion` ConfigVersion rejects 2.x for a 1.x request)
2. Make `install(TARGETS ginkgo …)` block conditional on `NOT OGL_USE_EXTERNAL_GINKGO` — when external, only OGL needs installing

## Consequence

Cannot access Ginkgo 2.0 SYCL improvements (better BJ, ParIC, Multigrid)
without OGL migration work.

## Status

- GitHub Issue: [to be opened at hpsim/OGL]
- OGL branch survey: no `ginkgo-2.0` branch exists
- `precond_update` branch: large refactor (424 files, +7111/-34328) that
  *removes* the broken UpdateMatrixValue/get_local_solver paths — but is
  marked WIP ("WIP copy old block_ptrs" is the latest commit) and is a
  total restructuring, not a targeted Ginkgo 2.0 migration

## Workarounds

None directly available. Options:
1. Wait for OGL maintainers to migrate to Ginkgo 2.0 API
2. Contribute migration PR (significant effort — likely more breaks beyond
   the first 36 errors)
3. Continue with Ginkgo 1.10 (current viable option, with all the SYCL
   preconditioner limitations documented in findings/02, 05, 08)
