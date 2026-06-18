# Tomorrow — execute Plan C then Plan D

Everything is staged. Backups exist (`*.bak-20260618-prereuse` in the ginkgo-src +
OGL trees). Work the steps in order; commit after each green result.

## Plan C — AMG values-only reuse (~15–20% wall-clock, the GPU-side lever)
Runbook: [`../knowledge/amg-reuse-port-plan.md`](../knowledge/amg-reuse-port-plan.md) ·
Reference code: [`../findings/code/ginkgo-patches/`](../findings/code/ginkgo-patches/)

1. **Port the code** into `/opt/ogl-src/build/release/_deps/ginkgo-src/`:
   - `multigrid_level.hpp`: add `class UpdateMatrixValue` (4 lines, from reference 01).
   - `pgm.hpp`/`pgm.cpp`: inherit it; add 3 cache members (`coarse_partition_`,
     `coarse_imap_`, `off_diag_map_`); store them in `generate()` (~pgm.cpp:416–467);
     add `Pgm::update_matrix_value` (adapt reference 02 to develop's `coarse_imap`).
   - `multigrid.hpp`/`multigrid.cpp`: inherit it; add `Multigrid::update_matrix_value`
     (reference 03 ports nearly verbatim — `set_system_matrix`/`handle_list` exist).
   - OGL `Preconditioner.hpp:667`: fix `#ifdef GINKGO_WITH_OGL_EXTENSION` →
     `...EXTENSIONS` (plural — the naming bug).
2. **Build:** `bash scripts/c-prep/build-c.sh` (~1h; iterate on compile errors — they
   are the develop API drift, mostly in the distributed Pgm branch).
3. **Test:** `bash scripts/c-prep/test-c-caching.sh` — PASS = init_precond drops on
   `caching>0` AND iters stay ~12–15 (a 201-cap = stale reuse → fix value-refill or
   lower N). Demidov warning: full reuse is risky for N-S — values-only must be exact.
4. **Record:** commit the patch to `findings/code/ginkgo-patches/`; update
   `knowledge/performance-maps.md` with the MEASURED C result (replaces the projection).
   Same-session A/B only (clock jitter ±2–3%).

## Plan D — full-float solve (VRAM lever for 30–35M)
Runbook: [`../knowledge/full-float-port-plan.md`](../knowledge/full-float-port-plan.md)

5. After C is green: add a `solvePrecision single` keyword that converts
   `dist_A`/`dist_b`/`dist_x` to float in `lduLduBase::solve_multi_gpu_impl`, builds
   float CG + Schwarz, converts `x` back to double. OGL-only rebuild (fast — float
   Ginkgo types already compiled).
6. Test accuracy (7.1M double vs single — same fields/iters) → VRAM at 17.2M
   (~−35–40%) → the **34M mesh** on one card (np≤4).

## Rollback (either plan)
Restore the `*.bak-20260618-prereuse` backups + `cmake -B /opt/ogl-src/build/release
-DGINKGO_WITH_OGL_EXTENSIONS=OFF` + rebuild.

## Guardrails (from config-pitfalls.md)
- `source scripts/cr2605-shell.sh` before any np>1 run (CR 26.05 LD-switch).
- Verify `executor sycl;` (default is CPU `reference`).
- Per-phase timing before/after; expectations are AUDITED (~15–20% wall-clock for C,
  NOT 2× — see performance-maps.md projection).
