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
avoids it (`cacheAgglomeration`: keep hierarchy, refresh values). Eliminating
it would roughly **halve** the solve time — a bigger lever than full-float.

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

## To unlock the ~halving (highest-value roadmap item)
1. Build Ginkgo with the **OGL extension** that provides `UpdateMatrixValue`
   (hpsim's Ginkgo fork) — a Ginkgo rebuild.
2. Fix the OGL `#ifdef GINKGO_WITH_OGL_EXTENSION` → `...EXTENSIONS` naming
   mismatch (Preconditioner.hpp:667) and enable the CMake option.
3. Then `caching N` reuses the hierarchy WITH value updates (GAMG-style) →
   init_precond ~0 + convergence preserved → ~2× on the solve.

## Other communication notes
- **copy_x_back bandwidth 0.94 GB/s** — far below PCIe; the D2H is tiny/
  latency-bound (consistent with the known PCIe Gen1x1 xe-BMG downgrade). Small
  absolute cost (18 ms) so low priority, but flags the slow host link.
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
   init_precond ~0 → **~2× on the solve** (init_precond was ~55-59%).

Estimated payoff: solve ~halved → GPU would beat CPU GAMG by ~2× (vs the ~1.06×
we have now via single precision). Bigger than full-float.

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
