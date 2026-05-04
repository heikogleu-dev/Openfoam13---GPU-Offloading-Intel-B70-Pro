# Test: scaling=-1.0 for SPD preconditioners (ISAI, IC) — does NOT recover convergence

## Background

OGL's README states:

> "preconditioners like IC or (SPD) ISAI require positive values on the
> system matrix diagonal, thus in case of the pressure equation the
> complete system needs to be scaled by a factor of -1.0"

Earlier docs (findings/02, 05) had not tested `scaling -1.0` explicitly —
ISAI was reported as diverging and IC/ICT as `NotImplemented`/`DEVICE_LOST`,
both potentially attributable to the missing scaling.

This finding documents three direct tests with `scaling -1.0` applied.

## Setup

- Hardware: B70 Pro, 34M cells, np=8, scotch decomposition
- Stack: CR 26.05 (post-rollback), IGC 2.32.7, libOGL.so rebuilt with
  `GINKGO_JACOBI_FULL_OPTIMIZATIONS=ON` + `-fp-model=precise`
- p-block setting: `scaling -1.0;` plus the preconditioner under test

## Test 1: ISAI sparsityPower=1 + scaling=-1.0

```
ISAIsyclGKOCG: Solving for p, Initial residual = 1, Final residual = 8.681, No Iterations 201
ISAIsyclGKOCG: Solving for p, Initial residual = 0.815, Final residual = 0.522, No Iterations 201
ISAIsyclGKOCG: Solving for p, Initial residual = 0.735, Final residual = 0.494, No Iterations 201
```

Result: **diverges on first solve** (Final 8.68 > Initial 1.0); subsequent
solves stagnate at ~50 % reduction per 200 iterations. The scaling did
ensure positive diagonal but ISAI's approximate inverse is too inaccurate
for the 34M-cell pressure operator at sp=1.

## Test 2: ISAI sparsityPower=3 + scaling=-1.0

```
terminate called after throwing an instance of 'sycl::_V1::exception'
what(): Provided range and/or offset does not fit in int.
        Pass `-fno-sycl-id-queries-fit-in-int` to remove this limit.
```

Result: **crashes during preconditioner generate** before any p-solve
runs. With `sparsityPower=3` the ISAI sparsity pattern blows past the
SYCL 32-bit `int` range that DPC++ uses for kernel offsets/ranges by
default on a 34M-cell mesh.

Theoretically fixable by rebuilding OGL/Ginkgo with
`-fno-sycl-id-queries-fit-in-int`, but no guarantee that the resulting
ISAI(3) would converge — and the rebuild adds overhead to all kernels.
Not pursued in this session.

## Test 3: IC + scaling=-1.0

```
terminate called after throwing an instance of 'gko::NotImplemented'
[8 ranks all aborted identically]
```

Result: **unchanged from findings/05** — `gko::NotImplemented` thrown by
`sparselib_ic` (Ginkgo's IC kernel is not ported to the SYCL backend).
Scaling has no effect on this — it is an absent code path, not a
numerical issue.

## Verdict

`scaling -1.0` correctly addresses the diagonal-sign requirement, but it
does **not** recover working SPD-preconditioner support:

| Preconditioner | Without scaling | With scaling=-1.0 |
|---|---|---|
| ISAI sp=1 | diverges | **diverges (worse)** |
| ISAI sp=3 | not tested | **SYCL int-range overflow** |
| IC | `NotImplemented` | **`NotImplemented` (unchanged)** |
| ICT | `DEVICE_LOST` | not retested (same root cause expected) |

The OGL README's recommendation is technically correct for an idealised
Ginkgo SYCL stack, but the actual Ginkgo 1.10 SYCL implementation has
multiple gaps (NotImplemented, int-range overflow, divergence) that
`scaling` alone cannot patch.

Confirms the broader finding: **only `BJ maxBlockSize=1` is viable**
in Ginkgo 1.10 SYCL for this mesh size, regardless of scaling.

## What `scaling -1.0` IS Useful For

If/when Ginkgo 2.0 OGL migration finishes and IC/ICT/ISAI become
implementation-complete, `scaling -1.0` will be the correct fvSolution
setting for the OpenFOAM pressure equation. So this is documented as a
pre-requisite knob, not a current fix.
