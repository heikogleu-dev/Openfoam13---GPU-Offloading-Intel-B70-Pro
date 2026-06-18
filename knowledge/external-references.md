# External references — the prior art we build on

Cross-checks for our findings. **Search this + the other KB files before
doing fresh external research.**

## Preconditioner theory (why ILU/BJ lose, why AMG wins)

- **Saad, *Iterative Methods for Sparse Linear Systems* (2nd ed.)** —
  canonical: ILU(0)/Jacobi give a constant-factor κ improvement but keep
  the O(h⁻²) asymptotics → CG iterations ~O(h⁻¹) = O(N^(1/3)) in 3-D;
  multigrid is mesh-independent (O(1) iterations, O(N) work).
- "One-level PCG solvers deteriorate for medium-to-large problems while
  multigrid multi-level PCG is fast" — multiple AMG-PCG papers, e.g.
  [An efficient AMG-PCG solver](https://www.sciencedirect.com/science/article/abs/pii/S004578250200378X),
  [A fast AMG-PCG solver](https://www.sciencedirect.com/science/article/abs/pii/S0096300305010416).
  Validates our 7.1M result (ILU ~192 iters ≈ N^(1/3), GAMG 3–5).

## GPU OpenFOAM pressure-solve (the win = GPU algebraic multigrid)

- **NVIDIA AmgX for OpenFOAM** — GPU AMG preconditioner for the pressure
  PCG; up to **9× pressure-solve speedup** vs GAMG-PCG, ~2–3× overall
  (Amdahl). Confirms: the GPU win comes from **AMG on the GPU**, not ILU/BJ.
  [Martineau/NVIDIA OpenFOAM 2020/2021](https://wiki.openfoam.com/images/a/a4/OpenFOAM_2020_NVIDIA_Martineau.pdf),
  [OpenFOAM with GPU Solver Support](https://www.keysight.com/cae/sites/default/files/resource/other/2611/OpenFOAM_Conference_2021_Abstract_Martineau_NVIDIA.pdf).
- **petsc4Foam** (+ Hypre / AmgX / cuSPARSE) — maintained GPU plug-in path.
  [arXiv:2403.07882](https://arxiv.org/pdf/2403.07882),
  [NEXTFOAM petsc4Foam](https://blog.nextfoam.co.kr/2024/01/10/gpu-accelerated-openfoam-with-petsc4foam/).
- **RapidCFD** — GPU OpenFOAM fork, unmaintained since 2016 (OF 2.3.1).
- **OpenFOAM on GPUs, state of play (2026)** — [Hivenet overview](https://www.hivenet.com/post/openfoam-gpu-state-of-play).

## Ginkgo / OGL (our stack)

- **Olenik et al. 2024, "Towards a platform-portable linear algebra
  backend for OpenFOAM", Meccanica** —
  [doi:10.1007/s11012-024-01806-1](https://link.springer.com/article/10.1007/s11012-024-01806-1).
  OGL design; KIT guidance "2 MPI subdomains per GPU".
- **Olenik et al., "Towards Distributed Linear Solvers on GPUs using
  Ginkgo"** — [KIT/OpenFOAM 2022](https://www.keysight.com/cae/sites/default/files/resource/other/3280/30_Abstract_OpenFOAM_2022_Olenik_KIT.pdf).
- **Ginkgo has AMG / multigrid on Intel GPUs** (W-cycle, mixed precision,
  Max 1550 / Ponte Vecchio) — [Intel: Ginkgo & oneAPI](https://www.intel.com/content/www/us/en/developer/articles/technical/ginkgo-and-oneapi-accelerate-numerical-simulations.html),
  [Tsai et al., Porting Ginkgo to Intel GPUs, arXiv:2103.10116](https://arxiv.org/pdf/2103.10116).
  → This is exactly the GPU-AMG we want; the open question is VRAM-viability
  + effectiveness through OGL's distributed path on Battlemage.
- [hpsim/OGL](https://github.com/hpsim/OGL), [ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo).

## Our own upstream reports

- [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922)
  — CR ≥26.14 multi-process `zeInit` abort on BMG-G31.
- [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
  — `lower_trs`/`upper_trs` SYCL gap (merged 2026-06-02), unblocks ILU/IC apply.
- OGL status / patches → `../logs/ogl-greole-consolidated-status.md`,
  OGL issue #170.

## Tuning & deeper sources (2026-06-17 research round)

- **Tsai, Beams, Anzt, "Three-Precision Algebraic Multigrid on GPUs",
  FGCS 2023** — [doi/full text](https://www.sciencedirect.com/science/article/pii/S0167739X23002741),
  [OSTI PDF](https://www.osti.gov/servlets/purl/2581015). Ginkgo's own AMG on
  H100/MI250X/**Intel PVC**: PGM size-2 aggregation + weighted-Jacobi smoother;
  AMG-CG iteration counts 10–359 (problem-dependent); W-cycle +12–60%; DP-SP
  mixed precision; **skip FP16 for short-row pressure Laplacian on Intel**
  (subgroup {16,32} → FP16 SpMV can be slower). Best GPU smoother = scalar Jacobi.
- **Notay, "Aggregation-Based AMG", ETNA 37 (2010)** —
  [PDF](https://etna.ricam.oeaw.ac.at/vol.37.2010/pp123-146.dir/pp123-146.pdf):
  unsmoothed aggregation is **not grid-independent** without smoothed
  aggregation or K-cycle Krylov acceleration → explains our ~13-iter floor
  with Ginkgo PGM.
- **Olenik et al., Matrix Repartitioning, arXiv:2510.08536** —
  [PDF](https://arxiv.org/pdf/2510.08536): ranks-per-GPU tuning; naive MPI
  oversubscription = ~140× slowdown; repartitioning up to ~10× on A100.
- **petsc4Foam + Hypre BoomerAMG, PRACE WP294** —
  [PDF](https://prace-ri.eu/wp-content/uploads/WP294-PETSc4FOAM-A-Library-to-plug-in-PETSc-into-the-OpenFOAM-Framework.pdf):
  the canonical BoomerAMG pressure config (strong_threshold 0.7 for 3D, HMIS,
  ext+i); 64M cavity 128 nodes: AMG-CG 79 s vs GAMG-PCG 553 s (scalability win).
- **NVIDIA AmgX configs** — [stock JSON configs](https://github.com/NVIDIA/AMGX/tree/main/src/configs),
  [Poisson production config (barbagroup)](https://raw.githubusercontent.com/barbagroup/cloud-repro/master/examples/poisson/azure/amgx/config/poisson_solver.info):
  PCG + CLASSICAL AMG, V-cycle, PMIS, D2, block-Jacobi, dense-LU coarse.
- **The AMG-resetup gotcha (AmgX 2.5× slower than GAMG)** — SPUMA benchmark
  [arXiv:2512.22215](https://arxiv.org/html/2512.22215v1): pressure coefficients
  change each SIMPLE iter → AMG hierarchy can't be cached like GAMG.
- **Oliani/Polimi, GPU coupled OpenFOAM, arXiv:2403.07882** —
  [PDF](https://arxiv.org/pdf/2403.07882): cells/GPU win threshold (~1–5M min,
  >10M ideal); LDU→CSR cached; LA drops from ~80% to ~20% of step on GPU.

> **Conflation warning:** the "2.3× GAMG GPU speedup" sometimes seen online is
> **RapidCFD** (CUDA OpenFOAM fork) on Titan X, **not** OGL/Ginkgo. Both real
> OGL papers benchmark vs CPU **PCG**, not GAMG.

## Intel-platform fit & FP64 / mixed precision (2026-06-17 research round)

See [intel-platform-fit.md](intel-platform-fit.md) for the synthesis.

- **B70 FP64 (our own measurement):** clpeak in
  [Ginkgo #2013](https://github.com/ginkgo-project/ginkgo/issues/2013) →
  **1335 GFLOPS FP64, 530 GB/s** (~1:8). Battlemage FP64 is native-rate-limited
  on XVE units, not Alchemist-style emulation. Alchemist had no native FP64
  ([Tom's Hardware](https://www.tomshardware.com/news/intel-arc-will-not-support-fp64-hardware)).
  Compute-runtime FP64 policy / `cl_khr_fp64`:
  [intel/compute-runtime FAQ](https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md).
  B580 ~909 GFLOPS clpeak: [Phoronix](https://www.phoronix.com/review/intel-arc-b580-gpu-compute/3).
- **We are the only documented consumer-Battlemage OGL/Ginkgo OpenFOAM user** —
  our [OGL #170](https://github.com/hpsim/OGL/issues/170),
  [compute-runtime #922](https://github.com/intel/compute-runtime/issues/922),
  Ginkgo [#2013](https://github.com/ginkgo-project/ginkgo/issues/2013)/[#2015](https://github.com/ginkgo-project/ginkgo/issues/2015)/[#2018](https://github.com/ginkgo-project/ginkgo/issues/2018).
- **Classical RS-AMG merged to Ginkgo develop** (CPU-only, unreleased):
  [PR #1985](https://github.com/ginkgo-project/ginkgo/pull/1985),
  GPU-kernel draft [PR #2034](https://github.com/ginkgo-project/ginkgo/pull/2034).
  Distributed MG since v1.8.0: [PR #1269](https://github.com/ginkgo-project/ginkgo/pull/1269).
  Mixed-MG example: `ginkgo/examples/mixed-multigrid-solver`.
- **Mixed-precision iterative-solver consensus** (FP64 outer + FP32 precond
  keeps full accuracy): Carson & Higham 2018
  [SISC](https://epubs.siam.org/doi/10.1137/17M1140819);
  Bake/Carson/Ma 2025 [arXiv:2510.11379](https://arxiv.org/abs/2510.11379);
  Abdelfattah et al. 2021 survey [arXiv:2007.06674](https://arxiv.org/abs/2007.06674).
- **FP32 floor ~1e-6 (keep reductions in FP64):** Neko/Nekbone Chen et al. 2025
  [arXiv:2503.02134](https://arxiv.org/abs/2503.02134). **Pure FP32 fails for
  turbulent LES** → OpenFOAM keeps the solve FP64: Brogi et al. 2022
  [arXiv:2209.06105](https://arxiv.org/abs/2209.06105).
- **FP16/short-row verdict** (FP64→FP32 only ~2.33× at 7 nnz/row; index traffic
  dominates): Loe et al. 2021 [arXiv:2105.07544](https://arxiv.org/abs/2105.07544);
  Ginkgo-on-Intel subgroups {16,32}: Tsai/Cojean/Anzt 2021
  [arXiv:2103.10116](https://arxiv.org/abs/2103.10116); FP16 range/squeezing:
  Higham/Pranesh/Zounon 2019 [PDF](https://eprints.maths.manchester.ac.uk/2678/1/paper.pdf).
- **FluidX3D on Intel Arc = excellent** (FP32/FP16 LBM, B70 ~6750 MLUPs/s):
  [FluidX3D README](https://github.com/ProjectPhysX/FluidX3D),
  [Phoronix B70](https://www.phoronix.com/review/intel-arc-pro-b70/5). Contrast:
  FP64 OpenFOAM-via-OGL is experimental/unsupported on consumer Arc.

> **Retraction logged:** an earlier research pass hallucinated a citation
> (arXiv:2509.16081) for "OGL wires mixed precision only for Block-Jacobi" — that
> paper says no such thing; the claim rests on our own code reading (Preconditioner.hpp:172).

## Where our work is genuinely new

The fundamentals above are established. Our novel contribution is the
**Intel Arc Pro B70 (Battlemage) + SYCL/Ginkgo/OGL** combination: the CR
driver multi-process bug, the Ginkgo SYCL preconditioner bugs and their
2.0 fixes, the `find_blocks` distributed-path failure, and the first
measured BJ/ILU/ISAI/**Multigrid** numbers on this hardware.

## Research sources (2026-06-18 deep dive) — the prior art we build on

**AMG hierarchy reuse (the #1 lever):**
- NVIDIA AmgX `structure_reuse_levels` + open coarse-sparsity-skip: [AMGX#127](https://github.com/NVIDIA/AMGX/issues/127)
- AMGCL partial reuse (−40–200% setup): Demidov [arXiv:2108.02054](https://arxiv.org/abs/2108.02054)
- Trilinos MueLu `reuse: RAP` (3.5×, problem-dependent): [OSTI 1364816](https://www.osti.gov/servlets/purl/1364816)
- PETSc-GAMG `-pc_gamg_reuse_interpolation`: [PETSc docs](https://petsc.org/release/manualpages/PC/PCGAMGSetReuseInterpolation/)
- **Ginkgo has NO reuse API:** issues [#1681](https://github.com/ginkgo-project/ginkgo/issues/1681), [#1158](https://github.com/ginkgo-project/ginkgo/issues/1158)
- OpenFOAM `cacheAgglomeration` (caches map, re-sums coarse ops): OpenFOAM-dev GAMGSolver.C

**Mixed / single precision (validates our single-precision win):**
- Oo & Vogel 2020 (FP32 inner MG + FP64 IR = full accuracy, 2.5×): [arXiv:2007.07539](https://arxiv.org/abs/2007.07539)
- Brogi 2022 (OpenFOAM: pure-FP32 fails turbulence; keep LA FP64): [arXiv:2209.06105](https://arxiv.org/abs/2209.06105)
- Carson & Higham 2018 (3-precision IR theory): [10.1137/17M1140819](https://doi.org/10.1137/17M1140819)
- Cojean et al. 2024 (Ginkgo per-level MP-AMG beats AmgX 1.5×): [10.1177/10943420241268323](https://doi.org/10.1177/10943420241268323)
- Neko 2025 (FP32 PCG, FP64 reductions, 29–38%): [arXiv:2503.02134](https://arxiv.org/abs/2503.02134)

**Smoother (the #2 lever, deprioritised by our 72%-setup finding):**
- Lottes 2022, 4th-kind Chebyshev: [arXiv:2202.08830](https://arxiv.org/abs/2202.08830); D'Ambra 2025 [arXiv:2407.09848](https://arxiv.org/abs/2407.09848)

**OpenFOAM-on-GPU ecosystem:**
- AmgX gold standard (7–9× pressure): coupled study [arXiv:2403.07882](https://arxiv.org/abs/2403.07882)
- Intel/KIT benchmark OGL+Ginkgo-SYCL on Max: [oneAPI Ginkgo blog](https://oneapi.io/blog/ginkgo-and-oneapi-accelerate-numerical-simulations-on-intel-gpus/)
- AMD MI300A OpenMP offload (4× H100): [arXiv:2405.00436](https://arxiv.org/abs/2405.00436); SPUMA portable fork (no Intel): [arXiv:2512.22215](https://arxiv.org/abs/2512.22215)

> Our contribution = the **Intel-Battlemage + SYCL/Ginkgo/OGL** combination, not the
> linear-algebra fundamentals above. See [gpu-comparison.md](gpu-comparison.md) for hardware context.
