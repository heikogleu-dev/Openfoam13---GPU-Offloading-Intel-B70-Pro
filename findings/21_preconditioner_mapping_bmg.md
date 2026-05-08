# Finding 21: Systematic SYCL Preconditioner Mapping on BMG-G31

## Context

Battlemage G31 is a new hardware generation, not yet in the Ginkgo
CI matrix. This finding documents the first systematic SYCL preconditioner
sweep on this architecture across Ginkgo 1.10 and 1.11.

## Test Matrix — 34M cells, np=8, OpenFOAM Foundation 13

| # | Solver | Preconditioner | Ginkgo 1.10 | Ginkgo 1.11 |
|---|---|---|---|---|
| 1 | GKOCG    | BJ (maxBlockSize=1) | ✅ runs (53.5 s/step, 200-iter cap) | ✅ runs (53.2 s/step, 201-iter cap) |
| 2 | GKOCG    | BJ (maxBlockSize=2) | ❌ size_t underflow | ❌ same underflow (V1+V2 L0 adapter) |
| 3 | GKOCG    | ICT (= ParICT)      | ❌ SIGABRT in `add_candidates` | ❌ same SIGABRT |
| 4 | GKOCG    | ISAI sparsityPower=1 | ✅ runs, mathematically diverges | ✅ runs (102 s/step), still diverges |
| 5 | GKOCG    | Multigrid (PGM)     | ❌ OOM + diverges (1.10) | not retested |
| 6 | GKOCG    | Chebyshev           | n/a (not exposed by OGL) | ❌ "preconditioner not supported by OGL" |
| 7 | GKOGMRES | BJ (maxBlockSize=1) | not tested | ❌ DEVICE_LOST — VRAM OOM, not bug, see [Finding 22](22_vram_pressure_gmres_oom.md) |
| 8 | GKOGMRES | ISAI sparsityPower=1 | not tested | ❌ DEVICE_LOST — same VRAM cause |

## Key Findings

**Only working SYCL preconditioner combination on BMG-G31 across both
versions:** `GKOCG + BJ(maxBlockSize=1)`. Mathematically too weak for
34M-cell pressure system (always hits `maxIter=200` cap), but the only
one that runs without crash and stays within VRAM.

**Three distinct failure classes confirmed:**

1. **Software bug — integer underflow:**
   `dpcpp::jacobi::find_blocks` for `maxBlockSize > 1`
   ([Finding 02](02_bj_blocksize_int_underflow.md), [18](18_v2_adapter_ruled_out.md), [19](19_ginkgo_111_upgrade_bug_persists.md))

2. **Software bug — SIGABRT in factorization:**
   `par_ict_factorization::add_candidates`
   ([Finding 05](05_sycl_preconditioner_status.md), reconfirmed for 1.11)

3. **Hardware limit, not bug — VRAM OOM:**
   GKOGMRES on this mesh size hits dedicated VRAM ceiling
   ([Finding 22](22_vram_pressure_gmres_oom.md))

## OGL-Layer Limitation

OGL exposes only `none, BJ, ILU, ISAI, IC, Multigrid` as preconditioner
keywords. Ginkgo 1.11 has additional preconditioners (Chebyshev,
batch_jacobi, sor) that are not reachable through OGL — would require
extending `OGL/include/Preconditioner.hpp` keyword switch.

## Failure Mode Persistence — Cross-Version Summary

| Failure | 1.10 | 1.11 | Class |
|---|---|---|---|
| `find_blocks` underflow (BJ>1) | ✅ confirmed | ✅ bit-identical | software bug |
| `add_candidates` SIGABRT (ICT)  | ✅ confirmed | ✅ confirmed     | software bug |
| ISAI divergence (CG)            | ✅ confirmed | ✅ confirmed     | mathematical, not bug |
| Multigrid OOM/diverge           | ✅ confirmed | not retested    | (likely persistent) |
| GMRES DEVICE_LOST               | not tested  | ✅ confirmed     | hardware OOM |

**No SYCL preconditioner failure was fixed by the 1.10 → 1.11 upgrade.**

## Files

- [`logs/stufe4-ginkgo111/test1-ict.log`](../logs/stufe4-ginkgo111/test1-ict.log)
  — ICT crash trace (1.11)
- [`logs/stufe4-ginkgo111/t2b-cg-isai.log`](../logs/stufe4-ginkgo111/t2b-cg-isai.log)
  — CG+ISAI divergence trace (1.11)
- [`logs/stufe4-ginkgo111/test2-gmres-isai.log`](../logs/stufe4-ginkgo111/test2-gmres-isai.log)
  — GMRES+ISAI DEVICE_LOST (1.11)
- [`logs/stufe4-ginkgo111/t2d-gmres-bj1.log`](../logs/stufe4-ginkgo111/t2d-gmres-bj1.log)
  — GMRES+BJ(1) DEVICE_LOST isolation (1.11)
- [`logs/stufe4-ginkgo111/test3-chebyshev.log`](../logs/stufe4-ginkgo111/test3-chebyshev.log)
  — Chebyshev OGL-keyword rejection (1.11)
