# GPU-AMG reference configs from others (AmgX, Hypre, Ginkgo)

Proven production configurations for accelerating the OpenFOAM pressure
(Poisson) solve, gathered to guide tuning Ginkgo Multigrid on the B70.
All cited; see [external-references.md](external-references.md) for URLs.

## The pattern across all mature GPU-AMG pressure solvers

- **Outer solver: PCG** (pressure is SPD). FGMRES/GMRES only for
  non-symmetric/coupled systems.
- **AMG as a 1-cycle preconditioner** (`max_iters=1` inside PCG), V-cycle.
- **Classical (Ruge-Stüben) AMG** for the coarsening — this is what reaches
  ~3–10 iterations. (Ginkgo does **not** have classical AMG — only PGM
  aggregation — which is why our Ginkgo MG floors at ~13 iters.)
- **GPU-friendly smoother**: block-Jacobi / l1-Jacobi / Chebyshev — **never
  Gauss-Seidel/SOR/SSOR** (sequential, slow on GPU). Confirmed by our own
  sweep (SSOR: 5–7 iters but 127 s/step).
- **Few sweeps** (1 pre / 1 post); escalate to (2,2) only if needed.
- A real **coarse solver** (dense-LU or several-iteration CG), not 1× Jacobi.

## NVIDIA AmgX — pressure (SPD) config [verbatim from stock + production]

```
solver = PCG ; max_iters = 100 ; tolerance = 1e-6 ; norm = L2
preconditioner = AMG
  algorithm   = CLASSICAL          # Ruge-Stüben (Ginkgo lacks this)
  cycle       = V
  selector    = PMIS               # parallel coarsening (GPU)
  interpolator= D2
  strength    = AHAT ; strength_threshold = 0.25
  presweeps = 1 ; postsweeps = 1 ; coarsest_sweeps = 1
  smoother    = BLOCK_JACOBI ; relaxation_factor = 0.9
  coarse_solver = DENSE_LU_SOLVER ; dense_lu_num_rows = 128
  max_levels  = 50–100 ; min_coarse_rows = 2 ; max_iters = 1
```
Aggregation path (for coupled/non-SPD): `algorithm=AGGREGATION`,
`selector=SIZE_2`, smoother `MULTICOLOR_DILU`, `postsweeps=3`,
`coarse_solver=DENSE_LU_SOLVER`.

## Hypre BoomerAMG (petsc4Foam) — pressure config [PRACE WP294 + Hypre defaults]

```
ksp_type = cg ; pc_type = hypre ; pc_hypre_type = boomeramg
boomeramg_max_iter        = 1
boomeramg_strong_threshold= 0.7      # 3D CFD (Hypre default 0.25 is a 2D value!)
boomeramg_coarsen_type    = HMIS     # PMIS on GPU
boomeramg_interp_type     = ext+i
boomeramg_grid_sweeps_up/down = 1/1
boomeramg_agg_nl = 2 ; agg_num_paths = 1
boomeramg_P_max = 2 ; truncfactor = 0.2 ; max_levels = 25
# default smoother = l1-Gauss-Seidel (fwd down / bwd up to keep CG symmetry)
# cycle default V(1,1); W-cycle "not recommended, not scalable"
```
petsc4Foam result: 64M cavity, 128 nodes — PETSc-AMG-CG **79 s** vs
FOAM-GAMG-PCG **553 s** (GAMG plateaus at 16 nodes; AMG scales to 64).
LDU→CSR conversion ≈ 2% of solve. **At scale AMG wins via scalability**,
not per-iteration speed.

## Ginkgo Multigrid — recommended config (what we can actually set in OGL)

Ginkgo example `mixed-multigrid-preconditioned-solver` idiomatic config:
PGM (size-2, `deterministic=true`) + IR(Jacobi, ω=0.9, 1 sweep) pre-smoother
+ **coarsest = IR(Jacobi) 4 sweeps or CG+IC** + V-cycle, 1 cycle per CG apply.
Stronger variant (`...-customized`): **IC smoother, 2 sweeps** + CG+IC coarse.

OGL keys (`knowledge/ginkgo-ogl-stack.md`): `cycle` v/w/f, `smoother`
Jacobi/SOR/SSOR, `maxIterSmoother`, `coarseSolver` Jacobi/CG,
`maxIterCoarse`, `maxLevels`, `minCoarseRows`, `relaxationFactor`.

**Our tuned best (matches the principles):** V-cycle + Jacobi smoother +
**CG coarse-solver** (maxIterCoarse ~20) → ~13 iters, ~9 s/step (1.17× GAMG).
W-cycle also good (~15 iters, 12 s). Avoid SSOR (sequential). See
[preconditioners-and-gpu-cfd.md](preconditioners-and-gpu-cfd.md).

## Reported GPU-vs-CPU numbers (context for expectations)

| Stack | speedup | scope | baseline | hardware |
|---|---|---|---|---|
| AmgX | ~7–9× | pressure solve only | CPU GAMG-PCG | V100/A100 vs 40-core |
| AmgX | ~2–5× | overall (Amdahl) | CPU GAMG | A100 vs EPYC |
| AmgX | **2.5× slower** | when AMG re-setup dominates | CPU GAMG | 2 GPU (SPUMA) |
| OGL/Ginkgo | ~15× / ~4× / ~2× | full timestep | CPU **PCG** (not GAMG) | 8×MI100 / 4×A100 / 4×Intel Max1550 |
| Ginkgo AMG-CG | 10–359 iters | (problem-dependent) | — | H100/MI250X/Intel PVC |

Intel-GPU specifics (Tsai/Anzt, Intel PVC): subgroup sizes limited to
{16,32}; for short rows (pressure Laplacian ~7 nnz/row) FP16 SpMV can be
*slower* than FP32 → skip FP16; use **DP-SP mixed precision** (FP64 vectors,
FP32 coarse). Best smoother on GPU was scalar Jacobi for most problems.

## Practical knobs that matter (OGL)

- `ranksPerGPU` / decomposition: ~2 subdomains/GPU rule of thumb; **naive
  oversubscription can be catastrophic (~140× slowdown)** — but our np sweep
  showed np=8 best here (single GPU, host-side assembly benefits).
- `forceHostBuffer true` (no GPU-aware MPI) costs **25–50%** vs GPU-aware MPI.
- Pressure system is negative-definite → `scaling -1.0` needed for IC/SPD-ISAI.
- LDU→CSR: sparsity built once, only values updated per solve (≈2% overhead).
