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

`Multigrid` sub-options: `smoother` (Jacobi), `coarseSolver` (Jacobi/CG),
`cycle` (v/w/f), `maxLevels`, `maxIterSmoother`, `maxIterCoarse`,
`minCoarseRows`, `fixedCoarsening`. Also `multiLevelSchwarz`, `Distributed`.

`BJ` sub-option: `maxBlockSize`.

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
