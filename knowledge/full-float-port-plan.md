# Full-float solve — port plan (D, the VRAM lever for 30–35M)

**Goal:** run the *whole* GPU pressure-solve (matrix + vectors + CG + Schwarz) in
FP32, not just the preconditioner. Primary win = **VRAM** (FP32 matrix halves the
dominant storage → fits the 34M mesh on one 32 GB card). Secondary = ~5–10% extra
bandwidth. NOT a big speed lever (see [performance-maps.md](performance-maps.md)).
Do this **after C** (it's independent, but C is the bigger speed win).

## Current state
- The **preconditioner** is already FP32 in `precision single` (our patch).
- The **outer solve is still FP64**: `scalar`(=double) typedefs remain for the
  distributed matrix, vectors, CG, and Schwarz. These are what D converts.

## Scope — the `scalar`→float surface (OGL)
| File | symbol | line |
|---|---|---|
| `lduLduBase.hpp` | `dist_vec = Vector<scalar>`, `dist_mtx = Matrix<scalar,…>` | 134–135 |
| `lduLduBase.hpp` | `PersistentDistributedMatrix dist_A` | 224 |
| `lduLduBase.hpp` | `create_dist_solver(...)` call | 291 |
| `Preconditioner.hpp` | `cg = Cg<scalar>`, `ras = Schwarz<scalar,…>` | 25, 31 |
| `StoppingCriterion.hpp` | `dist_vec = Vector<scalar>` | 22 |
| `GKOlduBase.hpp` | `create_dist_solver` macro | 24 |

**Feasibility gate PASSED:** Ginkgo 2.0 distributed types are templated on ValueType
— `Vector<float>`, `Matrix<float,…>`, `Schwarz<float,…>`, `Cg<float>` all build
(no hard double in the interfaces).

## Recommended approach — contained per-solve conversion (NOT a global re-template)
In `lduLduBase::solve_multi_gpu_impl` (~line 224–291): keep the OpenFOAM-facing RHS
/ result in double, but inside the solve **convert `dist_A`, `dist_b`, `dist_x` to
float once**, build a **float CG + float Schwarz + (already-float) MG**, solve, then
**convert `x` back to double**. This keeps the outer SIMPLE loop + assembly FP64
(the Oo&Vogel-safe "FP32 inner, FP64 outer" pattern; Brogi: pure-FP32 fails
turbulence, but pressure-Poisson-only FP32 with FP64 RHS is fine). Gate behind a
keyword, e.g. `solvePrecision single` (default double) — mirrors the precond patch.
- Avoids a global `scalar→float` re-template (which would lose the double option).
- The MG precond float path already exists; D adds the float matrix/vector/CG/Schwarz.

## Build + test
1. Backups already exist (`*.bak-20260618-prereuse`).
2. Rebuild OGL only (no Ginkgo change needed — float types already compiled) → fast.
3. **Accuracy test:** 7.1M, compare `solvePrecision double` vs `single` — same
   converged fields (residuals), iter count should stay ~13 (Carson-Higham: FP64
   RHS + FP32 solve recovers accuracy for our κ). If iters creep or residual
   stalls, the conversion is wrong.
4. **VRAM test:** measure peak at 17.2M (expect ~−35–40% vs double) → then the
   **34M mesh** (`Testcase` full) on one card, np≤4.

## Alternative (accuracy-safe, if FP32 solve worries) — memory accessor
Ginkgo's `range_accessors.hpp` stores the matrix in FP32 but **computes in FP64**
(storage precision ≠ compute precision). Halves matrix VRAM with zero accuracy
change, less bandwidth bonus than full-float. Fallback if D's FP32 solve shows any
convergence issue at 34M.

## ★ Precise scope (2026-06-19) — the crux is device-ingestion conversion, not typedefs

Mapped the full surface. `scalar` = `Foam::scalar` (=double, OpenFOAM-global), NOT an
OGL typedef → cannot just swap. The re-template surface:
- `Preconditioner.hpp`: `cg=Cg<scalar>`, `ras=Schwarz<scalar>` → float (fcg/fpgm/fir already exist from the precision patch).
- `lduLduBase.hpp`: `dist_vec=Vector<scalar>`, `dist_mtx=Matrix<scalar>`, `PersistentVector<scalar>` (b/x), `create_dist_solver` call.
- `GKOlduBase.hpp`: `create_dist_solver`/`create_precond`/`create_default` (dist_vec).
- `StoppingCriterion.hpp`: `dist_vec=Vector<scalar>`.

**The hard part (the crux):** `HostMatrixWrapper` holds **double** LDU pointers and
`PersistentVector<T>` reads the **double** OpenFOAM source. So the device side going
float needs **double→float conversion in the host→device copy** — in
`src/MatrixWrapper/Distributed.cpp` (`create_impl`/`update`, the `device_matrix_data`
build) and `DevicePersistent/Vector.hpp` (the vector init). This is a deep change to
the core data ingestion, not a typedef swap.

**Effort:** multi-file re-template + ingestion-conversion logic + rebuild + **`ninja
install`** (!) + accuracy test (FP32-inner/FP64-outer; watch iter-creep) + VRAM test
(17.2M then 34M). Comparable to C but riskier (conversion in the ingestion path).
**Recommend: execute fresh** (like C) — not at the tail of a long session.

**Accuracy-safe alternative — memory accessor:** keep the solver FP64 (no CG/Schwarz/
vector re-template), store ONLY the matrix in FP32 via Ginkgo `range_accessors`
(storage≠compute). Halves matrix VRAM with zero accuracy change. **Caveat:** must
verify Ginkgo's distributed Csr SpMV supports a mixed storage/compute accessor (uncertain;
the research listed it as a *potential* alternative). If supported, it's simpler +
safer than full-float and may suffice for the VRAM goal.
