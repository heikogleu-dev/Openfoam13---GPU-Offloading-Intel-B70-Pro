# Full-float solve — port plan (D, the VRAM lever for 30–35M)

## ★★★ DONE + VALIDATED 2026-06-19 — 34M FITS ON ONE 32 GB B70

Full-float implemented as a **compile-time toggle** `OGL_FULL_FLOAT` (CMake option →
`-DOGL_GSCALAR_FLOAT`), default OFF = byte-identical double behaviour (regression
guard). Device value type `gscalar` (=float when ON) is decoupled from `Foam::scalar`
(host/OpenFOAM double); conversion happens ONLY at host↔device value boundaries
(matrix all-to-all ingestion + vector init/update/copy_back). OGL-only rebuild
(all float distributed `Vector/Matrix/Schwarz` + the C-AMG-reuse extension
`update_matrix_value`/`get_local_solver` were already instantiated in libginkgo —
verified via `nm` — so **no Ginkgo rebuild**). Patch: `findings/code/ogl-patches/full-float-stack.patch`.

**Same-session A/B (run-fullfloat.sh, precision single + caching, GKOCG/MG):**
| mesh | np | double VRAM | float VRAM | iters (float vs double) |
|---|---|---|---|---|
| 7.1M | 8 | 7.45 GB | 8.18 GB (+0.73) | identical (18,15,13,13,11,12,14…) |
| 17.2M | 16 | 16.97 GB | **12.77 GB (−25%)** | near-identical |
| 34M | 16 | OOM (>32, decke ~28-30M) | **28.06 GB → FITS** | 25,21,17,15,15,13… healthy |

- **Accuracy:** FP32 outer reproduces FP64 convergence (Carson-Higham: FP64 RHS/tol +
  FP32 solve). iters match double essentially exactly. **No creep, no stall.**
- **Perf (7.1M):** float p-solve ~10–15% faster, wall ~3% faster (bandwidth).
- **★ Perf (34M, measured 2026-06-19, np=16, precision single + caching 2):** steady
  **~37 s/step** (periodic ~41 s rebuild steps), GPU compute util **35.7 %**, copy 6.2 %,
  render 0 % → still **CPU-bound** (~64 % wall = U/k/omega + assembly + MPI). **CPU GAMG
  34M = ~35.7 s/step → GPU is ~3–4% SLOWER at 34M.** So 34M = **VRAM win only, NOT a
  speed win** (the 7–17M ~1.2–1.3× edge does NOT extend to 34M: GPU scales 18→37 s
  17.2→34M [2.06×] vs CPU GAMG 22.1→35.7 [1.6× sublinear]; AMG rebuild 4.6 s vs 1.18 s
  reuse drives it). **Lever (untested):** higher caching to thin the 34M rebuilds.
- **VRAM:** per-cell slope **halved** (double ~0.94 → float ~0.45 GB/Mcell on 7.1→17.2).
  The +0.73 GB at 7.1M is a fixed-offset artifact (crossover ~8.6M); float wins big at
  scale. **Ceiling raised ~28-30M (double) → ≥34M (float). GOAL MET.**
- **Caveat:** VRAM steepens 17.2→34M (also np 8→16 between the 7.1M and ≥17.2M points,
  so the small-mesh slope isn't apples-to-apples). 34M still fits with ~4 GB headroom.
- **Rollback:** installed libs backed up — `libOGL.so.D-fullfloat-20260619` (float),
  `libOGL.so.C-working-20260619` (double). Instant `cp` swap. Currently FLOAT installed.
- **Build:** `cmake -B build/release -DOGL_FULL_FLOAT=ON -DGINKGO_WITH_OGL_EXTENSIONS=ON
  && ninja -C build/release && ninja -C build/release install` (install is mandatory).
- **NEXT:** converged forces (Cd/Cz) A/B float vs double at a target mesh; consider
  per-cell VRAM re-measure at fixed np; 34M production run now unblocked.

---


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

## ★ Memory-accessor feasibility VERDICT (2026-06-19) — go full-float instead

Checked. Mechanism: NOT the `range_accessors` (those are CB-GMRES Krylov-basis +
the dense-vector reads in mixed apply). The real "FP32 matrix / FP64 solve" path =
**Matrix<float> + mixed-precision apply** — and distributed::Matrix DOES support it
(`mixed_precision_dispatch_real_complex`, core/distributed/matrix.cpp:453/516). So
Matrix<float> × Vector<double> is technically feasible. **But:**
1. `GINKGO_MIXED_PRECISION` is **OFF** in our build → mixed apply is conversion-based
   (per-SpMV double↔float temp conversion = overhead) → would need it ON = Ginkgo rebuild.
2. Still needs the same **matrix-ingestion double→float** conversion (the create_impl crux).
3. Saves **less VRAM** (matrix only, not the vectors).
4. Its only edge (FP64-accuracy) is **MOOT** — FP32 is already proven accurate here
   (single-precond MG converges in the same ~13 iters).

**→ Go full-float** (uniform FP32): saves more VRAM (matrix+vectors), no mixed-apply
overhead, no GINKGO_MIXED_PRECISION rebuild, same proven precision. Memory-accessor is
only a fallback if the vector/CG/Schwarz re-template proves problematic.

### Fresh-session execution order (full-float)
1. Backup the OGL files (`*.bak-fullfloat`). 2. Add OGL-local `gscalar=float` aliases.
3. Re-template the solve types: `dist_vec`/`dist_mtx` (lduLduBase), `cg`/`ras`
   (Preconditioner), `dist_vec` (StoppingCriterion), create_dist_solver (GKOlduBase) → gscalar.
4. **The crux:** matrix-ingestion double→float in `src/MatrixWrapper/Distributed.cpp`
   (create_impl/update device_matrix_data build) + `DevicePersistent/Vector.hpp` (vector
   init converts the double OpenFOAM source → float device). 5. `ninja && ninja install` (!).
6. Accuracy test 7.1M (double vs full-float, same fields/iters). 7. VRAM 17.2M (−35-40%) → 34M.

## ★ Code-level ingestion crux (2026-06-19) — confirmed deep, two conversion paths

Read the ingestion. The conversion (host double → device float) needs work in BOTH:
1. **Vector path** (`DevicePersistent/Vector.hpp`): `VectorInitFunctor<T>` /
   `PersistentVector<T>` conflate the **source pointer type** and the **device value
   type** (both `T`; internal `dist_vec=Vector<scalar>`). The OpenFOAM source is
   `double*`. → Either (a) convert the OpenFOAM double source → a float host buffer
   and pass `PersistentVector<float>` (simplest, small host copy per solve), or (b)
   refactor VectorInitFunctor to separate Tsrc(double) from Tdev(float) + convert in
   `communicate_values`. (a) is the contained approach.
2. **Matrix path** (`src/MatrixWrapper/Distributed.cpp` create_impl/update): builds
   `device_matrix_data<scalar,label>` from `HostMatrixWrapper` (double LDU). → build
   `device_matrix_data<float,label>` reading the double host values (typed copy
   converts). HostMatrixWrapper stays double (host); only the device data goes float.
3. Then the solve types (cg/ras/dist_vec/dist_mtx, create_dist_solver, StoppingCriterion)
   → float, and the Schwarz-wrapped MG is already float-capable (single patch).

**VRAM note:** must build the device matrix float *directly* (NOT double-then-convert)
— a device-side convert would hold double+float simultaneously = MORE VRAM, defeating
the goal. So the ingestion must produce float.

**Rollback secured (2026-06-19):** installed C-libs backed up
(`<lib>.C-working-20260619`, instant `cp` rollback), OGL sources `*.bak-fullfloat`,
git tag `c-amg-reuse-working` (=80f3d28). Always returnable to the working C state.
