# Ginkgo / OGL stack — bugs, build, configuration

## Ginkgo SYCL preconditioner bugs (all FIXED in Ginkgo 2.0, standalone)

Four previously-blocking SYCL bugs, confirmed fixed in Ginkgo 2.0
standalone (single-process, up to 36M rows — see `findings/26`):

| Bug | Symptom | Status in 2.0 |
|---|---|---|
| `find_blocks` size_t underflow | BJ(maxBlockSize>1): `allocate ... 18446744073709551615B` | fixed **standalone**; still bites in OGL distributed (below) |
| `add_candidates` SIGABRT | ICT generate | fixed |
| `lower_trs` NotImplemented | ILU/IC *apply* on SYCL | fixed by [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023) (oneMKL trsm, merged 2026-06-02) |
| Multigrid PGM OOM | large meshes | algorithm runs; VRAM-bound (see vram file) |

## The `find_blocks` bug is in OGL's DISTRIBUTED path, not the kernel

This is the one OGL-actionable, still-open bug.

| Path | Result (Ginkgo 2.0) |
|---|---|
| Standalone Csr + `Jacobi(maxBlockSize=2..16)`, single process, up to 36M rows | ✅ runs clean |
| OGL distributed `Schwarz` wrapper, np≥2, BJ(maxBlockSize>1) | ❌ abort at `Generate preconditioner BJ<double>` |

- **Mesh-size-independent:** confirmed it aborts at **both 34M and 7.1M**
  (and historically `log.bs2`/`log.bs8` from May, same failure). So larger
  blocks are not an escape hatch; BJ(>1) is simply unusable through OGL
  distributed.
- BJ(maxBlockSize=1) works (but is ≈ diagonal scaling — too weak).
- Trigger is how `distributed::preconditioner::Schwarz` feeds the per-rank
  shard into `find_blocks` (empty row-blocks / sparsity the standalone
  Poisson matrix doesn't produce).

## OGL build on the current Intel stack

**Two source patches** (OF Foundation 13 + Ginkgo 2.0):
```diff
# src/MatrixWrapper/HostMatrix.cpp:239   (OF13 API rename)
-  patch.nbrPatchID()
+  patch.nbrPatchIndex()
# include/OGL/Preconditioner.hpp:270      (Ginkgo 2.0 Ilu template form)
-  gko::preconditioner::Ilu<ir, ir>::build()
+  gko::preconditioner::Ilu<scalar, false, label>::build()
```

**Four build-environment adjustments** (not OGL bugs):
1. `cmake --preset release` (not debug) — `-O0` makes icpx 2026
   `sycl-post-link` blow past 38 GB RSS on Ginkgo's ~120 kernels.
2. `-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON` — else a system Ginkgo is
   imported and `install(TARGETS ginkgo)` fails.
3. Remove `-fsycl-device-lib=all` from the fetched Ginkgo
   `dpcpp/CMakeLists.txt` (flag removed in icpx 2026).
4. Add `foam-shim/` include path to `OGL_public_api` (OF13 dropped `fvCFD.H`).

## OGL preconditioner keywords (from `Preconditioner.hpp`)

Valid `preconditioner` names: **BJ, ILU, ILUT, IRILU, IC, ICT, ISAI,
GISAI, Multigrid, none**.

`Multigrid` sub-options (verified dev source — see
[ogl-ginkgo-config-reference.md](ogl-ginkgo-config-reference.md) for the full,
corrected table): `type` (Schwarz/Distributed), `cycle` (v/w/f), `smoother`
(Jacobi/SOR/SSOR), `coarseSolver` (Jacobi/CG), `maxIterCoarse`, `relTolCoarse`,
`maxIterSmoother`, `relaxationFactor`, `maxLevels` (src default **20**, not 9),
`minCoarseRows` (src default **64000**, not 10).
⚠️ CORRECTION: `multiLevelSchwarz` and `fixedCoarsening` are **BJ-only** options
(`wrap_multi_level_schwarz`), **NOT** Multigrid sub-options (the MG path always
uses PGM). `precision` for Multigrid is our local patch only (upstream has it on BJ).

`BJ` sub-option: `maxBlockSize`. **Full key reference:
[ogl-ginkgo-config-reference.md](ogl-ginkgo-config-reference.md).**

## fvSolution GKOCG block (working template)

```
p
{
    solver              GKOCG;
    executor            sycl;
    matrixFormat        Csr;
    preconditioner      { preconditioner  ILU; }   // or Multigrid, BJ+maxBlockSize, ...
    forceHostBuffer     true;
    ranksPerGPU         <np>;        // must equal mpirun -np
    tolerance           1e-6;
    relTol              0.01;
    maxIter             200;
    adaptMinIter        true;
    relaxationFactor    0.8;
    updateInitGuess     true;
    evalFrequency       10;
}
```

## OGL gotchas
- **OGL requires `-parallel`.** Serial `foamRun` aborts:
  `Only parallel runs are supported for OGL` (ExecutorHandler.hpp:31). And
  `mpirun -np 1 ... -parallel` also aborts (OpenFOAM `UPstream::init`
  rejects `-parallel` with a single rank). → **minimum is np=2.**
- `ranksPerGPU` must match `mpirun -np`.
- Multi-rank needs CR 26.05 (see intel-compute-runtime file).

## Upstream status & roadmap (researched 2026-06-17)

- **`find_blocks` underflow (BJ>1): NOVEL, never confirmed by a maintainer.**
  Tracked only in our own issues ([Ginkgo #2013](https://github.com/ginkgo-project/ginkgo/issues/2013),
  [#2018](https://github.com/ginkgo-project/ginkgo/issues/2018),
  [OGL #170](https://github.com/hpsim/OGL/issues/170)). **Likely workaround:**
  Ginkgo docs call `find_blocks` "merely a heuristic" — passing explicit
  `block_pointers` bypasses it; may sidestep the underflow (untested). Best
  upstream-contribution candidate (minimal repro + the `size_t(-1)` arithmetic).
- **Classical Ruge-Stüben AMG just landed on Ginkgo `develop`**
  ([PR #1985](https://github.com/ginkgo-project/ginkgo/pull/1985), 2026-06-15)
  — **CPU reference only; GPU kernels are a draft ([#2034](https://github.com/ginkgo-project/ginkgo/pull/2034)),
  no SYCL yet, unreleased.** This is the real fix for the ~13-iter MG floor
  (classical AMG converges in few iters like GAMG) — but not usable on the B70
  yet. Smoothed aggregation: still none. PMIS "on the maintainer's list".
- **Distributed multigrid exists in Ginkgo since v1.8.0 (2024)** — only the
  *OGL integration* of it is open. Caveat: PGM aggregation is local-per-rank.
- **Mixed-precision multigrid is exampled + works** (Ginkgo
  `mixed-multigrid-solver` instantiates `Pgm<float>`; bugfix PR #1663). No
  SYCL-specific mixed-MG bug reported → an OGL DP-SP patch is feasible.
- **Ginkgo SYCL backend = DPC++/icpx-only** ([#2008](https://github.com/ginkgo-project/ginkgo/issues/2008));
  validated on data-center Max/PVC, consumer Arc untested. On SYCL in practice:
  point-Jacobi + CG/BiCGStab/GMRES solid; BJ(>1)/IC/ILU/ICT/ISAI-at-scale/PGM
  unreliable on Battlemage (our [#2015](https://github.com/ginkgo-project/ginkgo/issues/2015)).
- **OGL is actively developed** (last commit 2026-05-09) but no tagged release
  since v0.5.4 (2024); README documents only AMD MI100, no Intel guidance.
- SYCL build caveat: may need `-DCMAKE_CXX_FLAGS=-ffp-model=precise` (IEEE-754
  differences in Intel SYCL compilers) — relevant for mixed-precision results.

## Mixed-precision multigrid WIRED INTO OGL (2026-06-18, our patch)

`findings/code/ogl-patches/mixed-precision-multigrid.patch` adds a `precision`
keyword to the OGL Multigrid preconditioner block (was BJ-only before):
- `double` (default) — unchanged
- `mixed` — DP-SP: double finest level + float coarse levels (heterogeneous
  `with_mg_level(pgm_double, fpgm_float)` + `with_level_selector`)
- `single` — all-float MG preconditioner (`Pgm<float>`, float smoother + float CG coarse)

Validated 7.1M np=8 (CG-coarse, same ~12 iters all modes = no accuracy penalty):
| precision | VRAM | s/step |
|---|---|---|
| double | 11.4 GB | 8.67 |
| mixed | 10.4 GB (−9%) | 8.16 (−6%) |
| **single** | **8.5 GB (−25%)** | **7.95 (−8%)** |

`single` is the winner: −25% VRAM + −8% wall-clock, and it makes GPU-MG beat
CPU-GAMG (~8.5–9.3) even at 7.1M. VRAM ceiling ~20M→~25M. 34M still needs the
full-float *solve* (the double CG matrix remains) — next step.
