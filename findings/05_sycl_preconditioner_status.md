# Finding 05: SYCL Preconditioner Status (Corrected May 2026)

## Update — 2026-06-02: `lower_trs`/`upper_trs` Gap Closed Upstream

[ginkgo-project/ginkgo#2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
("Add triangular solver support for dpcpp" by @nbeams) was **merged into
Ginkgo `develop` on 2026-06-02 14:48 UTC** via oneMKL trsm. This closes
the apply-side gap (`dpcpp/solver/lower_trs_kernels.dp.cpp:43: generate
is not implemented`) that blocked every IC/ILU-family preconditioner
documented below.

Minimum oneAPI requirement was bumped to 2024.1 (we run 2026.0 — OK).

## Update — 2026-06-09: ILU actually reaches generate-phase on BMG-G31

After patching PR #168 with the two minimal fixes documented in
[Finding 24](24_pr168_patched_ilu_first_test.md) (`nbrPatchIndex` and
`Ilu<scalar, false, label>` template form), we built OGL against Ginkgo
2.0 `develop`. The `ILU` code path now reaches
`Preconditioner.hpp:219: Generate preconditioner ILU` — confirming that
the `lower_trs` `NotImplemented` is gone.

ILU then crashes with `DEVICE_LOST` during a `Csr::convert_to(Coo)`
inside ILU setup. VRAM behaviour: long stable plateau at ~26.5 GB
during setup, then a sharp spike past the 32 GB ceiling at the
format-conversion step. Theoretical bedarf is only ~15 GB (p + L + U +
Krylov + workspace) so the OOM is **partly hardware tightness, partly
a Ginkgo conversion-path that allocates a full Coo copy alongside the
source Csr**. Same pattern as
[Finding 22 (GMRES OOM)](22_vram_pressure_gmres_oom.md) — see
[Finding 24](24_pr168_patched_ilu_first_test.md) for the full
theoretical-vs-measured breakdown.

ILU on smaller meshes (≤8M cells) is the obvious next test to isolate
whether the algorithm itself works correctly given enough VRAM
headroom.

Issue [#2015](https://github.com/ginkgo-project/ginkgo/issues/2015)
remains **open** — `BJ find_blocks` underflow (Finding 02) and ISAI
int32 overflow on 34M cells are separate items.

---

## Important Correction (after KIT/Ginkgo team feedback, issue #2013)

Earlier versions of this document conflated two different algorithm
families. The Ginkgo team clarified:

| Algorithm Family | Type | Where it can fail in SYCL |
|---|---|---|
| **Classic IC / ILU** | Sequential, sparselib-based (cuSPARSE / hipSPARSE) | factorization itself (`sparselib_ic` not in SYCL) |
| **ParIC / ParILU** (Anzt 2018) | Iterative parallel | factorization runs, but **triangular-solve apply** missing in SYCL |
| **ParICT / ParILUT** | Parallel + dynamic sparsity refinement | SYCL implementation has internal bugs (`add_candidates`) |

Tsai 2023 refers to the **ParIC / ParILU** family, NOT the classic IC/ILU.
We were originally wrong to claim a "discrepancy" with the paper — the
factorization side is real. **The gap is in the triangular-solve apply
side that any IC/ILU-style preconditioner needs at runtime.**

Per Ginkgo team:
> "There is a branch where @yhmtsai is working on getting direct
> factorizations working for SYCL, but we have run into issues on some
> Intel hardware (the A770, for example) that we haven't yet figured out."

So direct factorization on SYCL is in development, with known A770 issues.
B70 Pro (Battlemage) status unknown — possibly different bugs.

## OGL Preconditioner-Name Mapping

OGL accepts six preconditioner names. Internally they map to Ginkgo
classes as follows (see `OGL/Preconditioner.hpp`):

| OGL `preconditioner` | Internal Ginkgo class | Algorithm category |
|---|---|---|
| `none` | identity | — |
| `BJ` | `gko::preconditioner::Jacobi` | Block-Jacobi |
| `IC` | `gko::factorization::Ic` (classic) | sparselib-based |
| `ICT` | `gko::factorization::ParIct` | parallel + threshold |
| `ILU` | `gko::factorization::ParIlu` | parallel iterative |
| `ISAI` | `gko::preconditioner::Isai<spd>` | sparse approximate inverse |
| `Multigrid` | `gko::solver::Multigrid` (PGM) | algebraic multigrid |

The names `ParIc` / `ParILU` are **not** valid OGL keywords — when used
they cause:
```
OGL does not support the preconditioner: ParIc
Valid Choices: none, BJ, ILU, ISAI, IC, Multigrid
```

So to test ParIc family, use OGL's `ICT` (= ParIct) or `ILU` (= ParIlu).
This was a non-obvious naming mismatch.

## Tested Preconditioners on B70 Pro / Ginkgo 1.10 SYCL (May 2026, post-rollback)

| OGL Name | Maps to | Status | Failure mode |
|---|---|---|---|
| `BJ maxBlockSize=1` | Jacobi (point) | ✅ Runs | Never converges — too weak (always 200-iter cap) |
| `BJ maxBlockSize > 1` | Jacobi (block) | ❌ OOM | SYCL workspace allocation O(N × BS²) |
| `IC` | `gko::Ic` (classic) | ❌ NotImplemented | `sparselib_ic` not in SYCL backend |
| `IC` + `scaling=-1.0` | (same) | ❌ NotImplemented | unchanged — `sparselib_ic` is absent code, scaling does not help |
| `ILU` | `ParIlu` factorisation + `Ilu` apply | ❌ NotImplemented in apply | `dpcpp/solver/lower_trs_kernels.dp.cpp:43: generate is not implemented` |
| `ICT` | `ParIct` factorisation + `Ic` apply | ❌ Crash | `par_ict_factorization::add_candidates` SIGABRT (or DEVICE_LOST in earlier driver) |
| `ICT` + `scaling=-1.0` | (same) | ❌ Crash (different path) | SIGABRT in `add_candidates`, scaling exposed a different code path |
| `ISAI sparsityPower=1` | `Isai<spd>` | ✅ Runs | Diverges (final residual > initial) |
| `ISAI sparsityPower=1` + `scaling=-1.0` | (same) | ✅ Runs | Still diverges (worse than without scaling) |
| `ISAI sparsityPower=3` + `scaling=-1.0` | (same) | ❌ Crash | `range/offset does not fit in int` — SYCL int32 overflow on 34M-cell sparsity pattern |
| `Multigrid` (PGM) | `solver::Multigrid` | ❌ OOM + Diverge | PGM coarsening OOM during `generate_local`; first solve also diverges |

## Bottom Line

**Only `BJ maxBlockSize=1` is stable on Ginkgo 1.10 SYCL distributed.** The
single working preconditioner is mathematically too weak for a 34M-cell
CFD pressure system (point-Jacobi never converges to `relTol=0.01`).

Every stronger option fails for one of three reasons:
1. **Not implemented in SYCL** (classic IC, lower/upper trs, ISAI sp>1)
2. **SYCL kernel bug** (ParIct add_candidates, BJ block workspace)
3. **Numerical divergence** (ISAI for the pressure operator, Multigrid first solve)

## What `scaling -1.0` IS Useful For

If/when the implementation gaps close, `scaling -1.0` is the correct
fvSolution setting for the OpenFOAM pressure equation (which has a
negative-definite system matrix by construction). We have it documented as
a pre-requisite knob, not a current fix — it is a no-op against any of the
implementation gaps documented above.

## Future Outlook

Per the KIT/Ginkgo team, there is active SYCL development including
`@yhmtsai`'s direct-factorization branch. The most realistic path to a
working strong preconditioner on Battlemage is:

1. Direct factorization (classic IC/ILU) lands in main Ginkgo + SYCL kernels — ⏳ in progress (@yhmtsai)
2. OGL adds `ParIc`/`ParIlu` keyword mapping (not currently present) — ⏳ pending
3. `lower_trs`/`upper_trs` SYCL kernels implemented for the apply side — ✅ **DONE 2026-06-02 ([PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023))**
4. ParIct `add_candidates` kernel bug fixed — ⏳ pending

Item 3 unblocks ILU/IC/ParIlu *apply*; items 1, 2, 4 still required for a
fully usable strong-preconditioner pipeline on this stack. **Status:
1 of 4 done.**
