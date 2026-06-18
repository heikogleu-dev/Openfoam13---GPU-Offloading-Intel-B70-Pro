# OGL + Ginkgo configuration reference (fvSolution keys, Multigrid, preconditioners, env, build flags)

> Source-grounded, CITED reference for tuning the OpenFOAM GPU **pressure** solve
> via OGL (hpsim/OGL) + Ginkgo on the Intel Arc Pro B70. Researched 2026-06-18.
> Cross-checked against upstream source + our empirical findings; conflicts flagged.

## 0. Which branch / version are we actually on? (read first — resolves most conflicts)

- **Our build = OGL `dev`-branch layout** (`include/OGL/*.hpp`), specifically the
  **PR #168** tree, built against **Ginkgo `develop`** (what we loosely call
  "Ginkgo 2.0" — there is **no tagged `v2.0.0`**; Ginkgo CHANGELOG ends at 1.11.0
  + an "Unreleased"/`INTERFACE_CHANGE.2.md` section). Confirmed by Finding 24
  (`include/OGL/Preconditioner.hpp:270`) and our patch file path.
- OGL has **two materially different layouts**:
  - `main` — old OpenFOAM-style `.H/.C` (`Preconditioner/Preconditioner.H`,
    `Solver/CG/GKOCG.H`). **Fewer keys**, `executor "sycl"` does **not** match
    (only `dpcpp`), and **GISAI / ILUT / IRILU are absent**.
  - `dev` (repo default) — refactored `include/OGL/*.hpp`. Adds `fuse`,
    `splitMPIComm`, `writeGlobal`, `precision` (BJ), `multiLevelSchwarz`,
    solver `GKOPipeCG`, accepts `executor sycl`, and **has GISAI/ILUT/IRILU**.
- **All line numbers below refer to the `dev` branch** unless noted, because that
  is our build. The README documents a mix and is often **stale** (see §6).
- ⚠️ "Ginkgo 2.0" everywhere in our KB/MEMORY should be read as "Ginkgo develop
  snapshot" (OGL pins it via `GINKGO_CHECKOUT_VERSION`; KIT maintenance forks
  `ogl_0600_gko190`=1.10, `ogl_0600_gko110`=1.11 also exist — Findings 19/20).

Primary sources: `https://github.com/hpsim/OGL/blob/dev/include/OGL/Preconditioner.hpp`,
`https://github.com/hpsim/OGL/blob/dev/include/OGL/lduLduBase.hpp`,
`https://github.com/hpsim/OGL/blob/dev/include/OGL/DevicePersistent/ExecutorHandler.hpp`,
`https://github.com/hpsim/OGL/blob/dev/README.md`,
`https://github.com/hpsim/OGL/blob/main/CMakeLists.txt`,
`https://github.com/ginkgo-project/ginkgo/blob/develop/CMakeLists.txt`,
`https://github.com/ginkgo-project/ginkgo/blob/develop/INSTALL.md`,
`https://github.com/ginkgo-project/ginkgo/blob/develop/include/ginkgo/core/solver/multigrid.hpp`.

---

## 1. fvSolution solver-block keys

### 1a. `solver` (TypeName registrations)

| Value | Note | dev source |
|---|---|---|
| **GKOCG** | CG (use for SPD **pressure**) | `Solver/CG.hpp` |
| GKOPipeCG | pipelined CG (fewer global reductions) — **dev only** | `Solver/PipeCG.hpp` |
| GKOBiCGStab | non-symmetric; **internally doubles `maxIter`** | `Solver/BiCGStab.hpp` |
| GKOGMRES | non-symmetric; VRAM-heavy (Krylov basis) — DEVICE_LOST/OOM at 34M (Finding 22) | `Solver/GMRES.hpp` |
| GKOIR | iterative refinement | `Solver/IR.hpp` |
| GKOMultigrid | MG as a *solver* (distinct from MG *preconditioner*) | `Solver/Multigrid.hpp` |

**Pressure rec: `GKOCG`.** GKOPipeCG unlikely to help here — with `forceHostBuffer`
the MPI Allreduce wait dominates (Finding 16), and pipelining targets exactly that
reduction. Source: `https://github.com/hpsim/OGL/tree/dev/include/OGL/Solver`.

### 1b. Core keys (all read via `lookupOrDefault` in `lduLduBase.hpp` / `StoppingCriterion`)

| Key | What it does | Values / **default** | B70 GKOCG rec. | Perf / VRAM |
|---|---|---|---|---|
| `executor` | Ginkgo backend | `reference`,`omp`,`cuda`,`hip`,`dpcpp`,**`sycl`** (dev) / **`reference`** | **`sycl`** | — |
| `matrixFormat` | sparse format | `Coo`,`Csr`,`Ell` (`Hybrid` = README only, **dead code**) / **`Coo`** | **`Csr`** | Csr = lowest mem + fastest SpMV for FV stencils; Coo default is slower |
| `ranksPerGPU` | MPI ranks repartitioned onto one GPU (device_id = rank/ranksPerGPU) | label / **1** | **= `mpirun -np`** (single GPU) | more ranks ⇒ more host-assembly parallelism (we are assembly/transfer-bound, ~46% CCS) |
| `forceHostBuffer` | copy to host before MPI (no GPU-aware MPI) | Switch / **false** | **`true`** (mandatory on xe) | adds host copies (~30% BCS); required for correctness |
| `updateRHS` | re-copy RHS to device each solve | Switch / **true** | `true` | RHS changes every solve |
| `updateInitGuess` | re-copy initial guess each solve | Switch / **false** | `true` | small copy |
| `updateSysMatrix` | re-copy system matrix each solve | Switch / **true** | `true` (SIMPLE matrix changes); `false` only if matrix constant | `false` saves one H→D matrix copy/solve |
| `scaling` | scale whole system by factor | scalar / **1.0** | `1.0` (CG); **`-1.0`** if using IC / SPD-ISAI (positive diagonal req.) | negligible |
| `tolerance` | absolute (normalized) residual tol | scalar / **1e-6** | `1e-6` (or `1e-5` loose) | tighter ⇒ more iters |
| `relTol` | relative tol (0 disables) | scalar / **1e-6** | `0.01` (matches CPU GAMG fairness) | — |
| `maxIter` | iteration cap | label / **1000** (×2 for GKOBiCGStab) | `200` (or `80` loose) | cap on solve cost |
| `minIter` | iteration floor | label / **0** | `0` | — |
| `adaptMinIter` | minIter = relaxationFactor×prevIters + adapt eval freq | Switch / **true** | `true` | skips early norm evals ⇒ faster |
| `relaxationFactor` | multiplier for adaptMinIter | scalar / **0.6** (src) — README says 0.8 | `0.8` | — |
| `evalFrequency` | check residual every N iters | label / **1** | `10` | fewer norm evals ⇒ faster (norm eval has cost) |
| `verbose` | log verbosity | 0–2 / **0** | `1` diag, `0` prod | log overhead |
| `export` | dump matrix+RHS to `.mtx` | Switch / **false** | `false` | huge disk I/O if on |
| `splitMPIComm` | split host/device communicator (**dev**) | Switch / **true** | `true` (default) | no measurable effect with forceHostBuffer (Finding 16) |
| `fuse` | fuse submatrices into one distributed matrix iface (**dev**) | Switch / **true** | `true` | — |
| `writeGlobal` | global indices when exporting `.mtx` (**dev**) | Switch / **false** (src; README false; one report true) | `false` | export-only |
| `MPIxRankOffload` | MPI×rank offload placement (**dev**) | bool / **false** | leave default | — |
| `debug` | write PID to `/tmp/mpi_debug_<rank>.pid`, sleep 20s (attach gdb) | Switch / **false** | `false` | n/a |

Also present but rarely tuned (in `StoppingCriterion`): `resNormEval` (0.1),
`normEvalLimit` (100) — adaptive-frequency internals; leave default.

⚠️ **`splitComm` vs `splitMPIComm`:** the **README** documents `splitComm`, but the
**source reads `splitMPIComm`**. Our Finding 16 tested `splitComm false` → "no
effect" — almost certainly because that key name is **silently ignored** (a no-op);
the real toggle is `splitMPIComm`. **Re-test with `splitMPIComm false` if revisiting.**

⚠️ **`regenerate` — does NOT exist** anywhere in OGL `dev` source (nor `main`, nor
README). It is not a recognized key. Solver/matrix re-copy is governed by
`updateRHS`/`updateInitGuess`/`updateSysMatrix`; preconditioner reuse by
`caching`/`preconditionerCaching` (§3).

⚠️ `executor "sycl"` works **only on dev**; on `main` only `dpcpp` matches (`sycl`
falls through to an empty executor pointer). Our use of `sycl` confirms we're on dev.

---

## 2. Ginkgo Multigrid preconditioner sub-options (the `preconditioner { preconditioner Multigrid; ... }` block)

Block at `Preconditioner.hpp:346-537` (dev). **Two `type` paths** (Schwarz default /
Distributed). All keys parsed at lines ~366-377.

| Key | What / Ginkgo mapping | Values / **default (src)** | B70 7M rec. | Note |
|---|---|---|---|---|
| `type` | Schwarz (local MG wrapped per-rank) vs Distributed (true distributed MG) | `Schwarz`/`Distributed` / **Schwarz** | **`Schwarz`** | only Schwarz validated on B70; Distributed hard-codes `max_iters=2` + has apparent CG/Jacobi label swap (likely upstream bug) |
| `cycle` | `.with_cycle()` | `v`/`w`/`f` (lowercase) / **v** | **`w`** | W: 14-17 iters @12.1s vs V: 55-68 @14s. ⚠️ bad value ⇒ UB (no else/default in src) |
| `smoother` | Jacobi→`Jacobi(max_block_size=1)`; SOR/SSOR→`Sor(symmetric=false/true)`, wrapped in Ir | `Jacobi`/`SOR`/`SSOR` / **Jacobi** | **`Jacobi`** | **NEVER SOR/SSOR** — sequential on GPU (SSOR×2 = 29s/26.5GB; combo = 127s) |
| `coarseSolver` | CG→`Cg` (BJ-preconditioned); Jacobi→`Ir` | `CG`/`Jacobi` / **Jacobi** | **`CG`** | single biggest wall-clock lever: 14s→9s |
| `maxIterCoarse` | coarse-solver iteration cap | label / **1** | **`20`** | default 1 makes "CG" coarse ≈ 1 Jacobi sweep — must raise |
| `relTolCoarse` | CG-coarse residual reduction (unused for Jacobi-coarse) | scalar / **1e-6** | `1e-6` | iteration cap usually binds first |
| `maxIterSmoother` | smoother (Ir) sweeps/level | label / **1** | **`1`** | ×3 sweeps = 17.2s (worse) |
| `relaxationFactor` | Ir relaxation ω for smoother/coarse | scalar / **0.9** | `0.9` | — |
| `maxLevels` | `.with_max_levels()` | label / **20** (README says 9 — stale) | `20` | — |
| `minCoarseRows` | `.with_min_coarse_rows()` | label / **64000** (README says 10 — stale) | `64000` (or 8000 + CG, ≈ tie) | — |

**Tuned best (matches our findings + AmgX/Hypre principles):** V or **W** cycle +
**Jacobi** smoother (1 sweep) + **CG** coarse (`maxIterCoarse ~20`) ⇒ ~12-14 iters,
~9 s/step (~1.17× CPU GAMG). Sources: dev `Preconditioner.hpp`,
`ginkgo .../solver/multigrid.hpp`; our `preconditioners-and-gpu-cfd.md` tuning map.

**`precision` (LOCAL PATCH only for Multigrid):** our
`findings/code/ogl-patches/mixed-precision-multigrid.patch` adds `precision`
(double/mixed/single) to the **Multigrid** block. Upstream `dev` has `precision`
**only on the BJ path** (`Preconditioner.hpp:172`) — the MG path is `scalar`-only.
**`single`** = −25% VRAM / −8% time, same ~12 iters (KB-validated). Not portable
without the patch; watch for overlap when rebasing onto upstream dev's BJ `precision`.

⚠️ **KB corrections (vs old `ginkgo-ogl-stack.md` list):**
- `multiLevelSchwarz` and `fixedCoarsening` are **NOT** Multigrid sub-options — they
  are **BJ-only** (`wrap_multi_level_schwarz`, lines 55-129). The MG path always uses
  `Pgm` and never reads them.
- `type` (Schwarz/Distributed) and `relTolCoarse` **were missing** from the KB list —
  both confirmed in source.

**Ginkgo MG capabilities NOT exposed by OGL** (can't set via fvSolution): independent
`post_smoother` (OGL hard-codes `post_uses_pre=true`), `mid_smoother`, K-cycle
(`kcycle_base`/`kcycle_rel_tol`), `solver_selector`, Pgm `deterministic`/
`max_iterations`/`max_unassigned_ratio`, and — critically — **classical Ruge-Stüben
/ smoothed aggregation don't exist in released Ginkgo at all** (only PGM). PGM
aggregation is *why* we floor at ~13 iters vs GAMG's 3-5.

---

## 3. Other OGL preconditioners (the `preconditioner` sub-dict — must be a sub-dict!)

**Syntax (Finding 03):** scalar `preconditioner BJ;` makes `e.isDict()==false`,
`d==dictionary::null`, and **all options silently fall back to defaults**. Always use
the sub-dict form: `preconditioner { preconditioner BJ; maxBlockSize 32; ... }`.

**Accepted keyword set (dev `Preconditioner.hpp`, verified line-by-line):**
`none(542), BJ(186), ILU(213), ILUT(228), IRILU(246), IC(275), ICT(291),
ISAI(318), GISAI(332), Multigrid(346)`. (NB: OGL's own fatal-error string and the
README under-report this set; **GISAI/ILUT/IRILU are real on dev** even though they
don't exist on `main`.)

| Keyword | Ginkgo class | Sub-options (default) | B70 SYCL status |
|---|---|---|---|
| `none` | — | — | valid no-op |
| `BJ` | `preconditioner::Jacobi` | `maxBlockSize`(**1**), `skipSorting`(true), `precision`(double, dev) | ✅ **only with maxBlockSize=1** (mathematically weak, caps at maxIter); **>1 ⇒ `find_blocks` size_t underflow** in OGL distributed (F02/F21) |
| `ILU` | `factorization::Ilu`→`preconditioner::Ilu` | `skipSorting`(true) | converges @7.1M (160-201 iters ~22-24s); **DEVICE_LOST OOM @34M** (CSR→COO, F24) |
| `ILUT` | `factorization::ParIlut`→`preconditioner::Ilu` | `skipSorting`(true) | **UNVERIFIED** on B70 (sibling of crashing ICT family — smoke-test) |
| `IRILU` | `preconditioner::Ilu<ir,ir>` (IR triangular solve, inner BJ(1)×5 hard-coded) | `skipSorting`(true) | **UNVERIFIED** — IR-trisolve may sidestep `lower_trs`/CSR→COO OOM; **good test candidate** |
| `IC` | `factorization::Ic`→`preconditioner::Ic` | `skipSorting`(true); needs `scaling -1.0` | ❌ `NotImplemented` (no SYCL IC kernel); scaling doesn't help (F15) |
| `ICT` | `factorization::ParIct`→`preconditioner::Ic` | `approximateSelect`(**true** on dev / false on main), `skipSorting`(true) | ❌ SIGABRT in `add_candidates` (F21/F05) |
| `ISAI` | `Isai<isai_type::spd>` | `sparsityPower`(**1**), `skipSorting`(true); needs `scaling -1.0` | ⚠️ runs, **diverges**; sp=3 ⇒ SYCL int-range overflow (F15) |
| `GISAI` | `Isai<isai_type::general>` (**dev only**) | `sparsityPower`(**1**), `skipSorting`(true) | **UNVERIFIED** on B70 (general variant; smoke-test) |

**Global preconditioner sub-options** (read for any preconditioner):
`skipSorting`(true), `caching`(**0**, label — reuse counter), `preconditionerCaching`(1,
master on/off), `approximateSelect`(ICT). Ginkgo's Jacobi `storage_optimization`/
`accuracy` and ParIlut `fill_in_limit`/`iterations` are **NOT exposed** by OGL.

**Net:** on B70 SYCL only **BJ(1)** (weak) and **ILU** (VRAM-bound) actually run; the
real pressure win is **Multigrid** (§2). IC/ICT/ISAI are broken/diverge on SYCL.
ILUT/IRILU/GISAI are dev-branch keywords we have **not yet smoke-tested** — IRILU is
the most interesting (IR-based trisolve).

---

## 4. Environment variables

| Var | Type | What / values | B70 rec. |
|---|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | runtime | `<backend>:<devices>`; backends `level_zero/opencl/cuda/hip/native_cpu`; devices `*/cpu/gpu/<num>/<num>.<num>` | **`level_zero:0`** (B70; iGPU = `:1`) |
| `SYCL_DEVICE_FILTER` | runtime | legacy filter | **DEPRECATED** — use ONEAPI_DEVICE_SELECTOR |
| `ZE_AFFINITY_MASK` | runtime | L0 (sub-)device exposure below SYCL | single-tile B70 ⇒ minor; per-rank pin if needed |
| `ZE_FLAT_DEVICE_HIERARCHY` | runtime | COMPOSITE/FLAT/COMBINED tile view | leave default |
| `SYCL_CACHE_PERSISTENT` | runtime | on-disk JIT cache (1/0, **off**) | **`1`** — Ginkgo JITs ~120 kernels; saves re-JIT each `foamRun` |
| `SYCL_CACHE_DIR` | runtime | cache root (default `$HOME/.cache/libsycl_cache`) | fast local FS |
| `SYCL_CACHE_TRACE` / `SYCL_UR_TRACE` | runtime | cache-hit / runtime-call diagnostics | one-shot debug only |
| `NEOReadDebugKeys` | runtime | unlock Intel NEO debug keys | not needed for normal solves |

**Ginkgo reads essentially NO runtime env vars** for device/executor/caching.
Executor is chosen at API level (driven by the `executor sycl` fvSolution key).
**`GINKGO_MIXED_PRECISION` and `GINKGO_DPCPP_SINGLE_MODE` are BUILD CMake flags, NOT
env vars** (common misconception). Ginkgo's only documented env reads are build path
hints (`CUDA_PATH`/`ROCM_PATH`/`HIP_PATH`) — none SYCL-relevant. Ginkgo profiling is
the **`ProfilerHook`** logger (NVTX/VTune/TAU), enabled **in C++**, not via env, and
OGL exposes no logger keyword — profile externally (vtune/unitrace).

Sources: intel/llvm `EnvironmentVariables.html`; Ginkgo `INSTALL.md`/`CMakeLists.txt`.

---

## 5. Build flags (all CMake, build-time)

| Flag | What / **default** | B70 rec. |
|---|---|---|
| `GINKGO_BUILD_SYCL` | SYCL/Intel kernels (needs dpcpp/icpx) / auto-on if SYCL compiler found (OGL re-declares default FALSE) | **ON** — the whole point |
| `GINKGO_BUILD_DPCPP` | legacy alias | DEPRECATED → use BUILD_SYCL |
| `GINKGO_MIXED_PRECISION` | instantiate **true** mixed-precision kernels (else conversion-based w/ temporaries) / **OFF** | **ON** if using the mixed/single MG patch — makes it perform optimally |
| `GINKGO_DPCPP_SINGLE_MODE` | **do not compile FP64 SYCL kernels** (all-FP32 backend) / **OFF** | **OFF** — B70 FP64 is strong (~1335 GFLOPS); CG matrix needs FP64. Precision wins come from OGL `precision` keyword, not this global switch |
| `GINKGO_JACOBI_FULL_OPTIMIZATIONS` | "all optimizations for the **CUDA** Jacobi algorithm" / **OFF** | ⚠️ **CUDA-scoped, likely no-op on SYCL** — we set it in some builds but no evidence it helps B70 |
| `GINKGO_FORCE_GPU_AWARE_MPI` | assume GPU-aware MPI ("may fail catastrophically" if not) / Ginkgo **OFF** but **OGL sets TRUE** ⚠️ | **explicitly `OFF`** — we use `forceHostBuffer` (host-staged); GPU-aware MPI not present on xe |
| `GINKGO_ENABLE_HALF` / `GINKGO_ENABLE_BFLOAT16` | FP16/bf16 value types / **ON** | OFF to shrink build; not used by CG pressure |
| `GINKGO_BUILD_REFERENCE` | reference CPU kernels / **ON** | keep ON |
| `GINKGO_BUILD_OMP` | OpenMP CPU kernels / OGL FALSE | optional CPU comparison executor |
| `GINKGO_WITH_OGL_EXTENSIONS` | **OGL-specific**; only adds `-DGINKGO_WITH_OGL_EXTENSIONS=1` to OGL target (does NOT patch Ginkgo) / **FALSE** | leave FALSE unless OGL code guards needed paths behind it — **trivial gating macro, not a perf knob** (name VERIFIED real) |
| `OGL_USE_EXTERNAL_GINKGO` | system vs bundled Ginkgo / **FALSE** | FALSE (bundled, pinned) — consistent with our `-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON` |
| `GINKGO_CHECKOUT_VERSION` | pins bundled Ginkgo commit | our actual Ginkgo version; bump deliberately for SYCL fixes |
| `-ffp-model=precise` (compiler) | IEEE-754 conformant; disables `-ffast-math` reassociation | ⚠️ **add it** — OGL ships `-ffast-math` which can perturb mixed-precision/CG determinism on icpx (effect **UNVERIFIED** on B70 — smoke-test) |

Our validated build-env adjustments (carry forward, all consistent with upstream):
`cmake --preset release` (avoid icpx sycl-post-link >38GB RSS at -O0),
`-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON`, drop `-fsycl-device-lib=all` from fetched
Ginkgo `dpcpp/CMakeLists.txt`, add `foam-shim/` include (OF13 dropped fvCFD.H), plus
the two code patches (`nbrPatchID→nbrPatchIndex`; Ilu template form).

**Recommended build line (B70 pressure):**
```
-DGINKGO_BUILD_SYCL=ON  -DGINKGO_MIXED_PRECISION=ON  -DGINKGO_DPCPP_SINGLE_MODE=OFF
-DGINKGO_FORCE_GPU_AWARE_MPI=OFF  -DGINKGO_BUILD_REFERENCE=ON  -DOGL_USE_EXTERNAL_GINKGO=FALSE
-DCMAKE_CXX_FLAGS="-ffp-model=precise"
# + --preset release, -DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON, drop -fsycl-device-lib=all
```

---

## 6. README-vs-source conflicts (source wins)

| Item | README | **Source (dev)** |
|---|---|---|
| `matrixFormat` default | Hybrid listed valid | **Coo** default; Hybrid = dead code; valid Coo/Csr/Ell |
| `relaxationFactor` default | 0.8 | **0.6** |
| Multigrid `maxLevels` | 9 | **20** |
| Multigrid `minCoarseRows` | 10 | **64000** |
| `caching` default | 1 | **0** |
| comm-split key name | `splitComm` | **`splitMPIComm`** (README key is a no-op) |

## 7. UNVERIFIED / open items
- `splitMPIComm false` never actually tested (Finding 16 tested README's `splitComm` = no-op).
- ILUT, IRILU, GISAI: dev keywords, **no B70 smoke test** — IRILU most promising.
- `-ffp-model=precise` and `GINKGO_FORCE_GPU_AWARE_MPI=OFF`: recommended but not yet
  empirically A/B'd on B70.
- `GINKGO_JACOBI_FULL_OPTIMIZATIONS` effect on SYCL: documented CUDA-only.
- Distributed-path Multigrid: untested on B70; hard-codes `max_iters=2` + apparent
  CG/Jacobi coarse-solver label swap (likely upstream bug).

## Cross-references
- `ginkgo-ogl-stack.md` (bugs/build/template), `preconditioners-and-gpu-cfd.md`
  (MG tuning map), `gpu-amg-reference-configs.md` (AmgX/Hypre patterns),
  Findings 02/03/15/16/21/22/24, `findings/code/ogl-patches/mixed-precision-multigrid.patch`.
