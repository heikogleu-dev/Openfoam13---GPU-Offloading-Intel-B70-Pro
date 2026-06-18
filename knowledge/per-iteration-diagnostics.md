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
