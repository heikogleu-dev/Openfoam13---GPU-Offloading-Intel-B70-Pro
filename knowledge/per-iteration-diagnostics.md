# Per-iteration / per-phase diagnostics (2026-06-18)

Goal: find where time/VRAM goes *inside* the pressure solve — and communication
(transfer/format/flow) potential. Sources: OGL's built-in `TIME_WITH_FIELDNAME`
phase timers (logged at `verbose>0`, already in run logs) + Ginkgo ProfilerHook.

## OGL phase breakdown per pressure solve (single precision, MG V+CG-coarse)

| phase | 7.1M np=8 | 17.2M np=8 | what it is |
|---|---|---|---|
| **init_precond** | **487 ms (~56%)** | **1650 ms (~59%)** | **AMG hierarchy rebuilt EVERY solve** |
| solve (CG loop) | 386 ms | 1143 ms | the actual iterations |
| call_update (H2D values) | ~per-update | 161 ms | LDU→CSR value refresh + host→device |
| call_init (first device setup) | — | 3438 ms (1×/step) | initial matrix/vector device init |
| copy_x_back (D2H) | — | 18 ms | solution back to host; **bandwidth only 0.94 GB/s** |

## ★ Top finding: AMG setup dominates, rebuilt every solve

**~55–59% of GPU solve time is the AMG hierarchy rebuild, repeated every
pressure solve.** This is the "AMG-resetup-per-step" gotcha — OpenFOAM GAMG
avoids it (`cacheAgglomeration`: keep hierarchy, refresh values). Eliminating it
would ~2× the *GPU pressure-solve*, but only **~15–20% of wall-clock** (the GPU
p-solve is ~40–48% of the step — audit-corrected, see bottom of file).

## The caching lever — present but NOT usable on our build

OGL has a `caching N` keyword (preconditioner reuse for N solves). Tested
7.1M single np=8:
| caching | init_precond | solve | iters | s/step |
|---|---|---|---|---|
| 0 (default) | 487 ms | 386 ms | 12–21 ✓ | 8.2 |
| 20 / 100 | 42 ms (−91%) | 4448 ms | **201 (cap)** from step 2 | 23 (worse) |

`caching N` *does* skip the rebuild (init_precond 487→42 ms) — but the
**value-update path is compiled out**: `gko::UpdateMatrixValue::update_matrix_value`
is **not in our libginkgo** (0 symbols), `GINKGO_WITH_OGL_EXTENSIONS=OFF`, and
the `#ifdef GINKGO_WITH_OGL_EXTENSION` (singular) **doesn't even match the CMake
define `...EXTENSIONS` (plural) — a naming bug**. So reuse keeps *stale* values
→ the AMG preconditioner becomes ineffective after the matrix changes (step 2+)
→ CG hits the 201-iter cap → net much slower. **Keep `caching 0` for now.**

## To unlock the GPU-solve speedup (highest-value GPU-side roadmap item)
1. Build Ginkgo with the **OGL extension** that provides `UpdateMatrixValue`
   (hpsim's Ginkgo fork) — a Ginkgo rebuild.
2. Fix the OGL `#ifdef GINKGO_WITH_OGL_EXTENSION` → `...EXTENSIONS` naming
   mismatch (Preconditioner.hpp:667) and enable the CMake option.
3. Then `caching N` reuses the hierarchy WITH value updates (GAMG-style) →
   init_precond ~0 + convergence preserved → ~2× on the solve.

## Other communication notes
- **copy_x_back bandwidth 0.94 GB/s** — a **software transfer-path** issue (tiny/
  latency-bound D2H: pageable host memory, map/unmap, small per-iteration copies),
  **NOT the PCIe link.** ★ Corrected 2026-06-18: the "PCIe Gen1×1" reading is an
  Intel-Arc switch-hierarchy **reporting artifact** — the real link (read at the
  parent bridge) is Gen5-class (~48–56 GB/s, clpeak on B70). Smoke-test with clpeak
  before chasing PCIe. Small absolute cost (18 ms) so low priority either way.
- `forceHostBuffer true` (no GPU-aware MPI) → the ~30% copy-engine in the maps.

## Within-solve breakdown (Ginkgo ProfilerHook nested summary, single, 0.5M)

Ignore "synchronize" rows (= the profiler's set_synchronization overhead); real
kernel time = "(self)".
- **MG preconditioner apply = ~96% of the solve** (CG outer vector ops are tiny).
- Within MG apply: **coarse-grid CG (maxIterCoarse=20) ≈ 50%** of the apply.
- **csr::spmv is the dominant kernel** (≈56% of the coarse CG) → bandwidth-bound
  → FP32/full-float helps.

→ **2nd lever: `maxIterCoarse` (default 20) looks oversized** — ~half the MG
apply is the coarse solve. Reducing it (5–10) should speed the solve with little
convergence cost. Cheap config test (unlike `caching`, this one should work).

## Root cause + fix path (refined 2026-06-18)

The `caching` value-update path needs `gko::UpdateMatrixValue` /
`Pgm/Multigrid::update_matrix_value` — an **hpsim Ginkgo-fork extension**
(branch `ogl_0600_gko110`, our old Ginkgo **1.11**; 82 symbols in that build).
It is **NOT in ginkgo-project develop** (our Ginkgo 2.0, which instead now has
the classical RS-AMG from PR #1985). The 2.0 migration dropped the extension →
caching reuse has no value-update → diverges (confirmed: even within-step reuse
fails because OGL's `call_update` refreshes matrix values every solve, so a
reused hierarchy is always stale vs the current matrix).

**Fix path (feasible, but a real project — the #1 performance lever, > full-float):**
1. Port the `update_matrix_value` extension from the `ogl_0600_gko110` 1.11 fork
   to the Ginkgo 2.0 source (UpdateMatrixValue interface + Pgm/Multigrid
   methods + the kernels that recompute Galerkin operators with new values —
   Ginkgo internals across a major version).
2. Rebuild Ginkgo 2.0 (heavy SYCL build, ~1h).
3. Fix the OGL `#ifdef GINKGO_WITH_OGL_EXTENSION` → `...EXTENSIONS` (Preconditioner.hpp:667)
   + enable the CMake option; rebuild OGL.
4. Then `caching N` reuses the hierarchy WITH value-update (GAMG-style) →
   init_precond drops ~91% (caching data: 487→42 ms) → **~2× on the GPU
   pressure-solve portion**.

Estimated payoff (audit-corrected, see bottom): ~2× on the GPU pressure-solve, but
that solve is only ~40–48% of the wall-clock step → **~15–20% wall-clock**, NOT 2×
over GAMG. Still the biggest GPU-side lever; bigger than full-float (which is a
VRAM lever for 34M, not a speed lever).

**maxIterCoarse is NOT a lever:** reducing 20→5 raised outer iterations
(12-21 → 27-41) enough to negate the cheaper coarse solve → net slightly worse.
20 is well-tuned. (Diagnostic logs: Testcase-half/log.lever-*, log.cache-*.)

## External validation + ranked levers (research 2026-06-18)

Our #1 finding (AMG hierarchy rebuilt every solve) is a **known, named problem**;
every major GPU-AMG library except Ginkgo already solves it by **reusing the
coarsening + transfer operators (P/R) and recomputing only the Galerkin coarse
operator values Ac = R·A·P** (sparsity fixed because the mesh is fixed):

| Library | reuse knob | reported |
|---|---|---|
| NVIDIA AmgX | `structure_reuse_levels=-1` (exactly our case) | gold standard; but even AmgX still recomputes coarse *sparsity* on resetup (open issue #127) |
| AMGCL (Demidov 2021) | partial reuse | **setup −40–200%, total −up to 20%**; **full reuse "hit or miss" → counterproductive for Navier-Stokes (iteration creep)** |
| Trilinos MueLu | `reuse: RAP` | setup 77→22 s (3.5×); but problem-dependent (16× *worse* on one MHD case) |
| PETSc-GAMG | `-pc_gamg_reuse_interpolation` | constant setup across timesteps |
| Hypre BoomerAMG | **none** | setup grows linearly with timesteps (relevant to our PETSc track) |
| OpenFOAM GAMG | `cacheAgglomeration` | caches only the *agglomeration map*; coarse matrices rebuilt by cheap **summation** of fine coefficients → that's why GAMG setup is cheap |

**This explains our caching=20/2 divergence empirically:** Demidov shows full
hierarchy reuse is *hit-or-miss and counterproductive for N-S* (iteration creep) —
exactly what we saw (201-iter cap after step 1). The correct path is **values-only
Galerkin refill, not stale full-reuse.**

**★ Key correction:** **Ginkgo has NO AMG-reuse API** (open issues #1681 "Reuse
pgm and multigrid", #1158 "values-only update, fixed sparsity"). `set_system_matrix`
only swaps the Krylov matrix, not the MG hierarchy. The hpsim 1.11 `update_matrix_value`
extension was an attempt at exactly this; it was dropped in the 2.0 migration. So a
**true values-only Galerkin refill in Ginkgo 2.0 would put us *ahead of the field***
(even AmgX leaves the coarse-sparsity-skip imperfect).

### Ranked levers to adopt (externally grounded)
1. **★★★ Values-only AMG hierarchy reuse** (cache PGM aggregation + Galerkin
   sparsity, refill Ac values per timestep). #1 bottleneck (~55–59%), fixed-mesh
   holds. Needs a Ginkgo-side patch (no API; port/rebuild the hpsim extension or
   implement against #1158). Interim: rebuild-frequency tuning (full-reuse is risky).
2. **★★ Fourth-kind Chebyshev polynomial smoother** (Lottes 2022, arXiv:2202.08830)
   — pure SpMV, no triangular solves, near-zero per-step setup, needs only a λmax
   estimate (reuse across fixed-pattern steps). Ideal for the B70's strong SpMV.
   Replace Ginkgo's default Ir+Jacobi smoother. (GPU framing is our inference; the
   paper is convergence theory.)
3. **★★ Mixed precision in the MG path** — DONE (our `precision single` patch);
   Ginkgo per-level MP-AMG exists upstream and beat AmgX 1.5× (Cojean 2024). Keep
   global reductions/residual in FP64 (Brogi 2022, Neko 2025, Oo&Vogel 2020).
4. **★ ParILUT + ISAI** cheaper-setup fallbacks (if AMG reuse stalls) — but only
   multigrid is mesh-independent for Poisson; ILU is not.

Sources: AMGX #127; AMGCL [arXiv:2108.02054](https://arxiv.org/abs/2108.02054);
MueLu [OSTI 1364816](https://www.osti.gov/servlets/purl/1364816); PETSc GAMG docs;
Ginkgo issues [#1681](https://github.com/ginkgo-project/ginkgo/issues/1681)/[#1158](https://github.com/ginkgo-project/ginkgo/issues/1158);
Oo&Vogel [arXiv:2007.07539](https://arxiv.org/abs/2007.07539);
Brogi [arXiv:2209.06105](https://arxiv.org/abs/2209.06105);
Cojean 2024 (IJHPCA 10.1177/10943420241268323); Lottes [arXiv:2202.08830](https://arxiv.org/abs/2202.08830).

## A — config-wins tested + refined per-phase split (2026-06-18)

Tested on 7.1M single MG np=8 (clean per-phase ms, OGL timers):
| config | init_precond | solve | call_update | call_init | s/step |
|---|---|---|---|---|---|
| baseline | 490 | 373 | 37 | 1415 (1×/step) | 7.97 |
| GINKGO_FORCE_GPU_AWARE_MPI=0 | 487 | 370 | 37 | 1415 | 7.95 (no change) |
| splitMPIComm false | 490 | 495 | 37 | 1410 | 8.2 (worse) |

**No config win** — both knobs already optimal: GPU-aware-MPI is a **no-op with
`forceHostBuffer`** (host buffers used regardless); `splitMPIComm true` (default)
beats false. (Confirms Finding 16's `splitComm` was the wrong key, and the right
one is already set best.)

**★ Refined finding — GPU time is SETUP-dominated, not solve-dominated:**
- Setup = call_init (1415, 1×/step) + init_precond (490×3) ≈ **2885 ms ≈ 72%**
- Solve (CG loop) = 373×3 ≈ **1119 ms ≈ 28%**

→ **Reprioritises away from the Chebyshev smoother (B):** it would only touch the
28% solve portion (and needs eigenvalue-bound estimation — Ginkgo's Chebyshev
has default foci {0,1}, no built-in estimator, no 4th-kind → risky). The 72%
setup (AMG rebuild + matrix init, every solve/step) is the real lever → **AMG
values-only reuse (C)** plus **call_init/matrix-structure reuse**. Data-driven:
do C before B.

## ★ AUDIT-CORRECTION (2026-06-18): honest wall-clock balance + C payoff

Earlier framing ("setup 72% of GPU time", C "~2×"/"halve the step") was **too
optimistic** — corrected here from phase-occurrence counts (A-baseline, 7.1M single
np=8, 5 steps):

| phase | per step | ms/step | note |
|---|---|---|---|
| init_precond | 3× (per p-solve) | 1470 | the AMG-rebuild — what C addresses |
| solve | 3× | 1119 | |
| call_update | ~8× | 311 | |
| call_init | **only 3× per 5 steps** (~0.6×/step) | ~849 avg, →~0 steady-state | matrix assembly, NOT every step |
| copy_x_back | 3× | 21 | |
| **GPU total** | | **~3770 (~2921 steady-state)** | |
| **wall-clock** | | **~7900** | |

**The GPU pressure-solve is only ~40–48% of the wall-clock step.** The other
~52–60% is **CPU** (U/k/omega DILU solves + p-matrix assembly + MPI) — GAMG runs
that same CPU work too, so the apples-to-apples comparison is fair, but it bounds
how much GPU-solve optimization can move wall-clock.

- `init_precond` ≈ **50–59% of the GPU pressure-solve time** (the old "55–59%" — true, narrow slice) but only **~18–21% of wall-clock**.
- **C (AMG values-only reuse)** removes ~⅔ of `init_precond` (the value-refill still costs) → **~10–15% wall-clock**, NOT ~2×. (May reach ~15–20% at 17.2M, where the GPU share of wall-clock is larger.)
- C is still worth doing (meaningful + ahead-of-field) — but expectations corrected.

## Full GPU offload — MEASURED negative result (2026-06-18)

Smoke (17.2M np16 single, 1 step): routing U/k/omega through OGL (GKOBiCGStab+BJ)
alongside p →
- **~3× slower** (s/step ~68 vs ~20.9 p-only), **compute-util collapses 38% → ~1%**.
- **Dominated by `call_init` ≈ 60 s/step** (n=18 — GPU matrix construction for 4
  equations, rebuilt every solve); a single k-solve took **10 s** (matrix build+transfer).
- VRAM +~50% (17→26 GB; see vram-and-mesh-scaling.md). It converges fine
  (U 3 iters, k 1, omega 2) — the problem is pure setup/transfer overhead, not math.

**Root cause:** U/k/omega are too cheap on CPU (1 DILU iter, ms) to amortize the
per-equation GPU matrix-build + H2D transfer (seconds each). You trade ms-CPU-solves
for s-GPU-setup. **Confirms the p-only architecture is correct** — the CPU is the
right place for the cheap turbulence/momentum solves.

**Connection to C:** same root cause as the AMG-rebuild (matrix/setup re-built every
solve). Full-offload would only be viable *with* matrix-setup caching (Plan C class),
and even then the cheap solves may never justify GPU offload. So the
"offload-everything to break the Amdahl ceiling" path is much harder than projected.
*(Caveat: 1-step smoke = cold setup; steady-state may amortize call_init somewhat,
but the 3× magnitude rules out a quick win.)*
