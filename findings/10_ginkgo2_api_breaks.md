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

---

## Upstream Issue Body (ready to file)

<!--
This is the prepared body for opening an issue at https://github.com/hpsim/OGL/issues/new
Use:
  gh issue create --repo hpsim/OGL --title "Ginkgo 2.0 API incompatibility: 3 breaking changes in Preconditioner.hpp" --body-file findings/10_ginkgo2_api_breaks.md
(use --body-file with the path to this consolidated file; or extract the section below)
-->

### Summary

OGL `dev` branch (current HEAD) is incompatible with Ginkgo 2.0 due to 3 API
breaking changes. This blocks users on Intel Arc/Battlemage hardware from
benefiting from Ginkgo 2.0's improved SYCL preconditioner support.

### Hardware / Software

- GPU: Intel Arc Pro B70 Pro (Battlemage G31, 32 GB, Xe2)
- OS: Ubuntu 26.04 LTS, Kernel 7.0.0-15
- OpenFOAM: Foundation 13
- Ginkgo: 2.0 develop (attempted), 1.10 (working)
- oneAPI: 2026.0.0, icpx

### Breaking Changes Found

#### 1. `gko::preconditioner::Ilu` template signature (Preconditioner.hpp ~line 270)
Ginkgo 2.0 changed the template parameters of `Ilu<ir, ir>`.
Affects: ICT preconditioner setup path.

#### 2. `Schwarz::get_local_solver()` removed (Preconditioner.hpp ~line 612)
Method no longer exists in Ginkgo 2.0.
Affects: Distributed mode with `preconditionerCaching > 0`.

#### 3. `gko::UpdateMatrixValue` interface removed (Preconditioner.hpp ~lines 614, 620)
This was the performance optimization for updating matrix values without
regenerating the preconditioner.
Affects: All cached preconditioner runs — core performance feature.

### Why This Matters for Battlemage/SYCL Users

With Ginkgo 1.10 SYCL on Intel Arc Pro B70 Pro, the following preconditioners
are broken or unusable:
- `BJ` `maxBlockSize > 1` → OOM (SYCL workspace allocation bug)
- `IC` / `ICT` → `gko::NotImplemented` (not ported to SYCL)
- `ICT` → `DEVICE_LOST` (GPU hardware hang)
- `Multigrid` → OOM in PGM coarsening + divergence
- `Hybrid` matrix format → "not supported in distributed mode"

Only `BJ` `maxBlockSize=1` (point-Jacobi) works, but it never converges for
34M-cell CFD pressure systems.

Ginkgo 2.0 claims improved SYCL preconditioner support and better distributed
multigrid — but we cannot test this due to the API incompatibility.

### Request

1. Is there a planned Ginkgo 2.0 migration branch for OGL?
2. Would a patch PR fixing these 3 API breaks be welcome?
3. Is the `precond_update` WIP branch related to this migration? (It removes
   the broken `UpdateMatrixValue`/`get_local_solver` paths but is marked
   "WIP" and is a 424-file refactor — not obviously a Ginkgo 2.0 target.)

### Reference

Full documentation of our testing session (B70 Pro + OpenFOAM + OGL):
https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro

All crashes are documented with stack traces and dmesg output.
