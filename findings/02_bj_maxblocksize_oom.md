# Bug: Ginkgo 1.10 SYCL Block-Jacobi OOM with maxBlockSize > 1

## Symptom

```
terminate called after throwing an instance of 'gko::AllocationError'
```

Occurs immediately during BJ preconditioner `generate()` step, before the
first CG iteration. All MPI ranks crash with `Signal: Aborted (6)`.

## Reproduction

```
preconditioner
{
    preconditioner  BJ;
    maxBlockSize    8;   // ANY value > 1 causes OOM
}
```

Tested on the 34M-cell vehicle aero case (4.25M cells/rank at np=8) on a
B70 Pro with 32 GB VRAM (29 GB usable).

## Root Cause

The SYCL/DPC++ implementation of Ginkgo 1.10's Block-Jacobi `generate()`
kernel allocates a temporary workspace that grows much faster than expected.
The `maxBlockSize=1` path is a special case (point-Jacobi) without the
workspace allocation.

Approximate workspace per rank, scaled with `(maxBlockSize)²`:

| maxBlockSize | Workspace/rank (4.25M cells) | Total (8 ranks on 1 GPU) | Status |
|---|---|---|---|
| 1 | ~0 (special path) | ~0 | ✅ works |
| 8 | ~2.2 GB | ~17 GB | ❌ OOM |
| 16 | ~8.7 GB | ~70 GB | ❌ OOM |
| 32 | ~34 GB | ~272 GB | ❌ OOM |

Even with maxBlockSize=8 and only 8 ranks (smallest tested combination),
the OOM hits — the memory model in Ginkgo's SYCL BJ generate is
fundamentally larger than what the docs suggest.

## Workaround

Use `maxBlockSize 1` (default — actually omit the option entirely).
The point-Jacobi path is mathematically weaker (no block coupling) but is
the only stable option for large meshes on SYCL backend.

Alternative if you need stronger preconditioning:
- Switch back to CPU GAMG (algorithmically much stronger anyway)
- Try Ginkgo 2.0 develop (`/opt/ginkgo`) — the SYCL BJ generate may be
  rewritten

## Note

The `GINKGO_JACOBI_FULL_OPTIMIZATIONS` CMake flag is **CUDA-only** and
does not affect the SYCL backend. There is no equivalent SYCL flag yet.

## Cross-Reference

This bug is the reason why our [maxBlockSize tests](../benchmarks/results.md)
all OOM'd. It limits OGL/SYCL on B70 Pro to point-Jacobi preconditioning,
which severely caps achievable convergence rate vs the Foundation default
GAMG (algebraic multigrid).
