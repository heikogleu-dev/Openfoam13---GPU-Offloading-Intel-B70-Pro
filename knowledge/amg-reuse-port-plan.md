# AMG values-only reuse — port plan (C, the init_precond lever)

**Goal:** stop rebuilding the AMG hierarchy every solve. Reuse the cached PGM
aggregation + transfer operators (P/R); recompute only the Galerkin coarse
operator **values** Ac = R·A·P (sparsity fixed because the mesh is fixed).
**Honest payoff (audited 2026-06-18):** `init_precond` is ~50–59% of the GPU
*pressure-solve* time but only **~18–21% of wall-clock** (the GPU p-solve is just
~40–48% of the step; CPU U/k/omega + assembly is the rest). Values-only reuse
removes ~⅔ of `init_precond` → **~10–15% wall-clock** (maybe ~15–20% at 17.2M).
NOT ~2× — earlier "halve the step" framing was an overstatement, corrected. Still
worth it (meaningful + ahead-of-field). See [per-iteration-diagnostics.md](per-iteration-diagnostics.md).

**Status (2026-06-18):** fully scoped, reference located, develop API drift mapped.
Remaining = the code surgery + a ~1h Ginkgo-SYCL rebuild + compile-fix loop +
test. Deliberately deferred to a fresh session (not done at the tail of a
marathon — high broken-build risk).

## The mechanism (from the hpsim Ginkgo-1.11 reference)
The 1.11 tree (kept locally at
`/opt/ogl-src/build/_deps/ginkgo-src.1.11.bak.20260522-2242/`) implements exactly
this via a `gko::UpdateMatrixValue` interface:
- **Interface** (`include/ginkgo/core/multigrid/multigrid_level.hpp`):
  ```cpp
  class UpdateMatrixValue {
  public:
      virtual void update_matrix_value(std::shared_ptr<const gko::LinOp>) = 0;
  };
  ```
- **`Pgm::update_matrix_value`** (`core/multigrid/pgm.cpp:499`): sets system_matrix_,
  then calls `generate_coarse(A_new, agg_)` reusing the cached aggregation `agg_`
  (+ `non_local_map_` in the distributed branch), then
  `set_multigrid_level(get_prolong_op(), coarse_new, get_restrict_op())`.
- **`Multigrid::update_matrix_value`** (`core/solver/multigrid.cpp:871`): mirrors
  `generate()` but, per level, calls `as<UpdateMatrixValue>(mg_level)->update_matrix_value(matrix)`
  instead of re-coarsening; regenerates the (cheap) smoothers + coarsest solver.

OGL already has the call site: `Preconditioner.hpp:685` calls
`gko::as<gko::UpdateMatrixValue>(local_solver)->update_matrix_value(...)`, gated
by the `caching`/`preconditionerCaching` keywords (see
[ogl-ginkgo-config-reference.md](ogl-ginkgo-config-reference.md)).

## develop API drift (the real adaptation work)
develop (`bec4d1ad9`, the active release tree
`/opt/ogl-src/build/release/_deps/ginkgo-src/`) differs from 1.11:

| 1.11 | develop | action |
|---|---|---|
| `Multigrid::set_system_matrix` | inherited from `EnableSolverBase` (solver_base.hpp:643) ✅ | use as-is |
| `Pgm` member `non_local_map_` | **gone** — distributed coarsening uses a local `coarse_imap` (`experimental::distributed::index_map`) + `off_diag_map` computed in `generate()` (pgm.cpp:416–445) | **must cache these** |
| 4-arg `generate_coarse(exec, csr, num_agg, agg)` | same (pgm.cpp:147) ✅ | local branch ports cleanly |
| 6-arg `generate_coarse(...non_local_map_)` | 6-arg takes `off_diag_map` + `coarse_imap.get_non_local_size()` (pgm.cpp:441) | rewrite distributed branch |

**Key surgery (distributed branch):** the value-refill must reuse what depends
only on sparsity/aggregation. In develop those are computed locally in
`generate()` and discarded, so:
1. Add Pgm members to cache: `coarse_partition_`, `coarse_imap_`, `off_diag_map_`
   (+ `agg_` already a member).
2. In `Pgm::generate()` (distributed branch, ~pgm.cpp:416–467) store them into the
   new members.
3. Write `Pgm::update_matrix_value` (develop): recompute only
   `generate_coarse(local_csr_NEW, num_agg, agg_)` and
   `generate_coarse(off_diag_csr_NEW, ..., agg_, coarse_imap_.get_non_local_size(), off_diag_map_)`,
   rebuild the coarse distributed Matrix from cached `coarse_imap_` + new csrs,
   `set_multigrid_level(get_prolong_op(), coarse, get_restrict_op())`.
   (Non-distributed branch ports almost verbatim from 1.11.)
4. `Multigrid::update_matrix_value` ports nearly verbatim (uses `set_system_matrix`,
   `handle_list`, the smoother lists, `mg_level_list_` — all present in develop).

## File-by-file checklist
1. `include/ginkgo/core/multigrid/multigrid_level.hpp` — add `class UpdateMatrixValue` (4 lines, after `namespace gko {`).
2. `include/ginkgo/core/multigrid/pgm.hpp` — `Pgm : ... public UpdateMatrixValue`; declare `void update_matrix_value(...) override;`; add the 3 cache members.
3. `core/multigrid/pgm.cpp` — store cache in `generate()`; add `Pgm::update_matrix_value`.
4. `include/ginkgo/core/solver/multigrid.hpp` — `Multigrid : ... public UpdateMatrixValue`; declare the method.
5. `core/solver/multigrid.cpp` — add `Multigrid::update_matrix_value` (port from 1.11:871).
6. **OGL** `include/OGL/Preconditioner.hpp:667,etc.` — fix `#ifdef GINKGO_WITH_OGL_EXTENSION` → `GINKGO_WITH_OGL_EXTENSIONS` (singular→plural bug) and ensure the macro is defined for the OGL compile.
7. Build: rebuild Ginkgo SYCL (the release tree) **then** OGL (~1h total). Watch VRAM ceiling unchanged (reuse saves time, not memory).

## Test plan
- `caching 1` (rebuild every other solve via the value-update path) and
  `caching N` on 7.1M single np=8; measure `init_precond` (should drop on cache
  hits) + iters (must NOT creep — if it does, the reuse is stale: lower N).
- Watch the Demidov warning: **full** hierarchy reuse is hit-or-miss for N-S
  (iteration creep). Values-only Galerkin refill should be safe; verify
  empirically with the diag-run harness + update [performance-maps.md](performance-maps.md).

## Why this is worth it
Every major GPU-AMG lib does this (AmgX `structure_reuse_levels`, MueLu RAP 3.5×,
PETSc-GAMG, AMGCL −40–200% setup). **Ginkgo mainline has no reuse API**
(issues #1681/#1158); even AmgX leaves the coarse-sparsity-skip imperfect → a clean
values-only Galerkin refill in Ginkgo 2.0 puts us ahead of the field. Reference
methods saved under `findings/code/ginkgo-patches/`.

## IMPLEMENTATION STATUS (2026-06-19) — ported + builds; cross-step refresh BUG

**Done:** the port is implemented (commit in `findings/code/ginkgo-patches/amg-reuse-2.0-port.patch`)
and **builds clean** (full SYCL build 2m35s; `update_matrix_value` symbol now in libginkgo;
`GINKGO_WITH_OGL_EXTENSIONS=ON`). 4 Ginkgo files + OGL `#ifdef` fix. The dev-API-drift fix
was a missing `Schwarz::get_local_solver()` getter (added). **Simpler than planned:** no new
Pgm cache members — the distributed coarse_imap/off_diag_map are recomputed from `agg_`; the
OGL Schwarz path only exercises the **non-distributed** branch (local block).

**Validated working:** caching path engages (init_precond drops 487→~230 avg on cache-hits);
**within-step reuse is correct** (iters 18→15→13, identical-matrix nNonOrth solves) — proves the
plumbing (OGL→Schwarz→MG→Pgm update) is wired right.

**THE BUG (open):** **cross-step reuse produces a broken preconditioner** — when the matrix
changes between timesteps, the refreshed preconditioner diverges: precision `double` → CG hits
the 201-iter cap; precision `single` → NaN residual (`OpenFOAMDistStoppingCriterion`). Pattern:
`18 15 13 13 [201 201 161 …]` — healthy until the first cross-step reuse.

**Ruled out (by reading, no rebuild):** stale `cache_.state` (apply uses `get_system_matrix()`
fresh, line 1152); stale matrix values (`updateSysMatrix` default true → dist_A_v current);
`set_multigrid_level`/dims; the apply reads `mg_level_list_`/smoothers/coarsest fresh each apply.
So my refresh *receives* the new values but yields a *broken* (not merely suboptimal) precond
— points to a correctness bug in the refreshed coarse op (generate_coarse(M_new, agg_)) and/or
the regenerated smoother/coarsest, NOT a no-op.

**Next (needs instrumentation = rebuild cycles):** add a checksum/norm log in
`Pgm::update_matrix_value` (input matrix + new coarse op) to confirm the new values reach
generate_coarse and the coarse op changes cross-step; bisect coarse-refresh vs smoother-regen
vs coarsest-regen (e.g. temporarily skip smoother/coarsest regen to see if the coarse alone is
the culprit). The double-precision 201 case is the cleaner debug target (no NaN noise).

**System state:** build kept (extension ON); `caching` unset = full-rebuild path = production
works normally. Backups at `*.bak-20260618-prereuse` for rollback.
