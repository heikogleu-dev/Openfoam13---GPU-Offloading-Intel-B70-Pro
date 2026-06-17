# Preconditioners & GPU CFD — the core finding

## TL;DR

For an elliptic **pressure** equation, the linear solver's cost is
dominated by the **preconditioner's convergence rate**, not by raw
hardware speed. One-level preconditioners (Block-Jacobi, ILU) need an
iteration count that **grows with mesh size**; algebraic multigrid (AMG,
e.g. OpenFOAM's GAMG) needs a **mesh-independent** number of iterations.
Therefore a GPU beats CPU GAMG **only** with a GPU-side algebraic
multigrid — not with BJ or ILU, no matter how they are tuned, decomposed,
or how much VRAM is available.

## The theory (established — Saad, *Iterative Methods for Sparse Linear Systems*)

For the 3-D Poisson/pressure operator on a mesh with spacing `h`
(N = total cells, `h ~ N^(-1/3)`):

| Method | condition number κ | CG iterations ~ √κ | mesh dependence |
|---|---|---|---|
| no preconditioner | O(h⁻²) | O(h⁻¹) = O(N^(1/3)) | grows |
| **ILU(0) / Block-Jacobi** | O(h⁻²) (smaller constant) | **O(N^(1/3))** | **still grows** |
| **Algebraic multigrid** | O(1) | **O(1)** | **independent** |

Key point: ILU(0) and BJ give a *constant-factor* improvement, **not an
asymptotic one** — the iteration count still scales like `N^(1/3)` in 3-D.
Multigrid is the only family that is mesh-independent (optimal O(N) work).

**External validation:** "the efficiency of common one-level PCG solvers
deteriorates for medium to very large size problems, while the multigrid
multi-level PCG method can be a fast solution method" — confirms the
one-level-vs-multigrid gap. See [external-references.md](external-references.md).

## Our measurements on the B70 (this stack)

### Half-resolution 7.1M-cell mesh, np=8, from uniform init, relTol 0.01

| Preconditioner | pressure CG iters/solve | s/step | peak VRAM | verdict |
|---|---|---|---|---|
| **GAMG (CPU, np=8)** | **3–5** | **~7.7 s** | host | reference (O(1), mesh-independent) |
| **Multigrid (Ginkgo, GPU)** | **55–101** | **~14 s** | 10.4 GB | ✅ **converges — best GPU option, ~1.8× GAMG** |
| ILU (GPU) | 160–201 | ~22–24 s | 11.0 GB | converges slowly; N^(1/3): 7.1M^⅓≈192 ✓ |
| ISAI (GPU) | 201 (caps) | ~15 s | 4.6 GB | too weak (never converges) |
| BJ(maxBlockSize=1) | 201 (caps) | — | — | too weak (≈ diagonal scaling) |
| ICT (GPU) | — | — | — | **crash** before solver (generate fails) |
| BJ(8), BJ(16) | — | — | — | **abort** in OGL distributed (`find_blocks`) |

**The key correction: not all preconditioners are too weak — Multigrid
works.** Ginkgo's algebraic multigrid converges in 55–101 iterations
(vs ILU's 160–201, vs BJ/ISAI capping at 201) at ~14 s/step — only ~1.8×
slower than CPU GAMG, and this is the **untuned default** (V-cycle, Jacobi
smoother, default levels). With a W-cycle, a stronger smoother, and more
levels it could plausibly close the gap. This is the GPU candidate worth
pursuing — exactly mirroring why NVIDIA AmgX (also GPU-AMG) beats GAMG.

**ILU notes (still useful):** the ILU iteration count N^(1/3)≈192 matches
the measured 160–201 — correct ILU(0) behaviour, not a misconfiguration.
Verified **not** an additive-Schwarz / rank artifact: a rank sweep gave
essentially identical iteration counts and only the VRAM and wall-clock
change with np:

| ILU np | iters (steady) | s/step | peak VRAM |
|---|---|---|---|
| 2 | 201/161 | ~32 s | 9.6 GB |
| 4 | 201/161 | ~26 s | 10.1 GB |
| 8 | 201/161 | ~23 s | 11.0 GB |
| 12 | 201/161 | ~23 s | 11.9 GB |

Iteration count is rank-independent; wall-clock improves to np=8 then
saturates (single GPU); VRAM rises slightly with np (Schwarz overlap).

### Consequence that matters

Because ILU is N^(1/3) and AMG is O(1), the **GPU-ILU disadvantage grows
with mesh size**:

| mesh | ILU iters ~N^(1/3) | GAMG iters |
|---|---|---|
| 7.1M | ~192 | 3–5 |
| 34M | ~324 | ~5 |

A bigger-VRAM card would let ILU *run* at 34M but it would lose to GAMG by
an even wider margin. **VRAM is not the limiter — preconditioner quality is.**

## Multigrid tuning map (7.1M, np=8) — the path toward parity

Tuning Ginkgo's Multigrid moved it from ~14 s/step to **~9 s/step**, i.e.
within ~1.17× of CPU GAMG (7.7 s) — on a mesh where the GPU is *under-fed*
(literature wants >10M cells/GPU; see below). Steady s/step = step2→3 delta.

| MG config | iters/solve | s/step | peak VRAM | note |
|---|---|---|---|---|
| 01 default (V / Jacobi / 1 sweep / Jacobi-coarse) | 55–68 | 14.0 | 11.4 GB | baseline |
| 02 **W-cycle** | 14–17 | 12.1 | 10.4 GB | big iter drop, cheap |
| 03 F-cycle | 27–36 | 13.0 | 11.5 GB | |
| 04 smoother sweeps ×3 | 33–40 | 17.2 | 10.8 GB | more smoothing costs more than it saves |
| 05 SSOR smoother ×2 | 20–25 | 29.0 | 26.5 GB | SSOR is sequential → slow + heavy |
| **06 CG coarse-solver** (maxIterCoarse 20) | **12–14** | **9.0** | 11.5 GB | **★ best wall-clock** |
| 07 combo (W + SSOR + CG-coarse) | **5–7** | 127 | 26.6 GB | fewest iters but SSOR kills wall-clock |
| 08 deep-coarse (minCoarseRows 8000 + CG) | 9–12 | 9.2 | 10.8 GB | ≈ tie with 06 |

Lessons (all match the literature, see [gpu-amg-reference-configs.md](gpu-amg-reference-configs.md)):
- **Fixing the coarse solver (Jacobi → a few-iter CG) is the single biggest
  wall-clock lever** (14→9 s). Default Jacobi coarse solve is weak.
- **W-cycle** sharply cuts iterations (55→15) at lower cost than extra smoothing.
- **SSOR/Gauss-Seidel are sequential on a GPU → avoid** — config 07 reaches
  GAMG-like 5–7 iters but at 127 s/step and 26.6 GB. Textbook GPU-AMG advice:
  use Jacobi / l1-Jacobi / Chebyshev smoothers, never Gauss-Seidel.
- **Why we floor at ~13 iters, not GAMG's 3–5:** Ginkgo only has **PGM**
  (unsmoothed size-2 aggregation) — no classical Ruge-Stüben and no smoothed
  aggregation. Aggregation AMG is documented as *not* grid-independent on its
  own. AmgX/Hypre reach few iters via **classical AMG**, which Ginkgo lacks.

Best working GPU config today: **Multigrid, V-cycle, Jacobi smoother
(1 sweep), CG coarse-solver (maxIterCoarse ~20)** → ~13 iters, ~9 s/step.

## The two caveats that reframe "beating GAMG"

1. **The fair CPU baseline in the literature is PCG, not GAMG.** Every
   published OGL/Ginkgo pressure speedup (Olenik et al., up to 15×) is vs CPU
   *diagonal-PCG*, **not** GAMG. Nobody has published OGL beating GAMG — we are
   holding the GPU to a higher bar than the field does. GAMG is an optimal
   multigrid *solver* with Gauss-Seidel/DIC smoothers (cheap on CPU, 3–5 iters).
2. **The pressure-specific killer: AMG setup can't be cached like GAMG.**
   Pressure-matrix coefficients change every SIMPLE iteration, so the AMG
   hierarchy must be re-built each timestep, whereas OpenFOAM GAMG reuses its
   hierarchy. In one published benchmark this alone made NVIDIA **AmgX 2.5×
   slower than GAMG**. This is a structural headwind for *any* GPU-AMG, not a
   B70/Ginkgo flaw.

## GPU is under-utilized at 7.1M — it's CPU-assembly/transfer-bound, not compute-bound

Measured B70 engine utilization during the tuned MG solve (np=8, 7.1M, via
fdinfo `drm-cycles-ccs`, no root — intel_gpu_top needs CAP_PERFMON):

| engine | busy | meaning |
|---|---|---|
| **CCS (compute)** | **~46%** | the SYCL/Ginkgo kernels — only half-busy |
| **BCS (copy)** | **~30%** | host↔device transfers (the `forceHostBuffer` cost) |
| RCS (render) | 0% | no graphics |

**The GPU compute engine is only ~46% busy** — this directly confirms the
solve is **CPU-assembly + transfer bound at this mesh size, not GPU-compute
bound.** That is *why* more MPI ranks make it faster (they parallelize the
host-side LDU→CSR assembly / value-update / MPI / residual that the GPU waits
on), and why the GPU is under-fed (matches the >10M-cells/GPU threshold).

Implications:
- **More ranks faster is expected here** (single GPU + CPU-heavy assembly) — it
  is *not* a sign of "CPU solving"; the pressure solve does run on the GPU, the
  GPU just spends half its time waiting on host work.
- **Mixed-precision's payoff is reframed:** halving GPU compute barely helps
  (compute isn't the bottleneck); the host→device matrix transfer stays double
  (OpenFOAM LDU is double). So mixed precision's concrete win is **VRAM** (GPU-
  internal hierarchy) → enabling bigger meshes — not raw speed at 7.1M.
- **Highest-value speed levers** (in order): (1) feed the GPU more cells so
  compute dominates and overhead amortizes; (2) GPU-aware MPI to kill the ~30%
  copy-engine cost (`forceHostBuffer` → direct device buffers, +25–50% per the
  literature); (3) reduce CPU-assembly cost / rank balance.

## Where the GPU actually wins: feed it more cells

Published thresholds: GPU pressure-AMG needs **~1–5M cells/GPU minimum, ideally
>10M cells/GPU** to beat CPU; below ~1M/GPU the CPU wins on overhead. Our 7.1M
single-GPU run is right at the *lower* edge — the GPU is under-fed, yet tuned
Multigrid is already ~1.17× GAMG. **The natural next test is a larger mesh
(~15–18M, the most that fits MG in 32 GB at ~1.6 GB/M cells) to feed the GPU
properly** — that is where parity/win is most likely.

## What the field already does (so we don't reinvent it)

GPU pressure-solve acceleration is a solved problem **on NVIDIA** via
GPU-side AMG:

- **NVIDIA AmgX** (external GPU solver for OpenFOAM): up to **9× speedup
  of the pressure solve** vs OpenFOAM GAMG-PCG; ~7× on the linear solver
  on 1×V100 vs 40-core CPU, but only **~2–3× overall** (Amdahl — pressure
  is one part of the step; the rest stays on CPU). The win comes from
  **AMG on the GPU**, exactly the preconditioner class BJ/ILU are not.
- **PETSc4FOAM** + Hypre / AmgX / cuSPARSE — the maintained plug-in path.
- **RapidCFD** — GPU OpenFOAM fork, unmaintained since 2016 (OpenFOAM 2.3.1).
- **Ginkgo has AMG** (multigrid, W-cycle, mixed precision) and runs it on
  Intel GPUs (Max 1550 / Ponte Vecchio) — so the algorithm exists in our
  stack; the open question is whether it is VRAM-viable and wired through
  OGL's *distributed* path on Battlemage.

**Our novel contribution** is therefore not "ILU is weak" (textbook) but:
making this work on **Intel Arc Pro B70 (Battlemage)** with the
**SYCL/Ginkgo/OGL** stack, and surfacing the **CR driver bug** and the
**Ginkgo SYCL preconditioner bugs** that block it. The path to a real
B70 GPU-CFD win is **Ginkgo Multigrid through OGL**, mirroring AmgX.

## Practical recommendation (current)

- Production pressure solving: **CPU GAMG** (np = all P+E cores).
- GPU: LLM inference + visualization, until Ginkgo Multigrid is
  VRAM-viable and effective through OGL's distributed path on the B70.

## Cross-references
- VRAM limits per preconditioner → [vram-and-mesh-scaling.md](vram-and-mesh-scaling.md)
- The `find_blocks` BJ>1 bug, ILU OOM mechanics → [ginkgo-ogl-stack.md](ginkgo-ogl-stack.md)
- Full run logs → `../findings/30_post_recovery_clean_multirank_perf.md`
