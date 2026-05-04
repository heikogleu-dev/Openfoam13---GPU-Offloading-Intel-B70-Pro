# Revision: GPU Kernel-Launch Latency is ~5.6 µs, NOT ~100 µs

## Summary

Earlier docs claimed Level Zero kernel launch latency on B70 Pro was
"~100 µs" — used as the primary explanation for why GPU CFD doesn't beat
CPU. **A direct measurement shows the actual sync latency is 5.6 µs**, with
async batching at 1.5 µs/launch. **The "100 µs" figure was wrong.**

This means kernel-launch overhead is **not** the dominant CFD bottleneck.

## Measurement (May 2026, post-update with CR 26.05 + IGC v2.32.7)

```
Device: Intel(R) Graphics [0xe223] (B70 Pro)

=== Synchronous (.wait() per launch) ===
       Trivial kernel + sync     1000 launches      5.6 µs/launch
       Trivial kernel + sync     5000 launches      5.6 µs/launch

=== Asynchronous (single sync at end) ===
        Trivial kernel async     1000 launches      1.5 µs/launch
        Trivial kernel async    10000 launches      1.5 µs/launch
```

Benchmark code: minimal `single_task` kernel that increments a single int,
1000 / 5000 / 10000 launches in a loop, measured end-to-end with
`std::chrono::high_resolution_clock`.

## Comparison

| Backend | Sync launch latency |
|---|---|
| CUDA typical | ~5 µs |
| **Level Zero on B70 Pro (measured)** | **5.6 µs** |
| OpenCL typical | ~20-30 µs |
| Earlier (wrong) docs | ~100 µs |

**B70 Pro / Level Zero is on par with CUDA for kernel launch latency.**

## Implication for the CFD Bottleneck Story

If kernel launch is 5.6 µs sync:

- 200 GKOCG iterations per p-solve × ~10 kernel launches per iter ×
  3 p-solves per timestep × 5.6 µs = **~33 ms/step kernel overhead**
- We measure **53 s/step** total
- → kernel-launch overhead is **0.06% of total time**, not the bottleneck

## Where the Time Actually Goes

Re-prioritized list of likely real bottlenecks for the 53 s/step that BJ(1)
takes:

1. **PCIe Host↔Device buffer copies** — `forceHostBuffer=true` is required
   in OGL (no GPU-aware MPI). Each MPI Allreduce/halo exchange copies
   through host. With 200 iterations each doing halo exchanges across
   8 ranks, that's 1600 PCIe round-trips per p-solve × ~10-20 ms each = many
   seconds per step.

2. **MPI Allreduce wait** — All ranks barrier at every CG inner product.
   With 8 ranks of slightly varying clocks, the slowest rank gates the rest.

3. **BJ(1) preconditioner ineffectiveness** — Solver always hits
   maxIter=200 cap. With a real preconditioner converging in 30-50 iters,
   we'd see 5-10× fewer kernel launches AND fewer PCIe round-trips.

The first two cannot be fixed without major OGL surgery (GPU-aware MPI).
The third is blocked by Ginkgo SYCL preconditioner bugs (findings/02, 05, 08).

## What This Changes in the Repo

- `README.md` claim "Level Zero kernel launch latency: ~100 µs" should
  be removed or corrected
- `conclusions.md` should be updated to reflect the new mental model:
  GPU is fine, but the OGL/Ginkgo distributed plumbing (host buffers +
  weak preconditioner) is what holds it back

## Possible Origin of the "100 µs" Figure

Likely confused two things:
- **Pure kernel launch:** ~5.6 µs (as measured here)
- **End-to-end GPU operation including OpenFOAM/OGL framework overhead**
  (parameter marshalling, error checks, OGL DevicePersistent state lookups,
  Ginkgo executor dispatch): could plausibly be ~100 µs
  
The framework overhead is real but is a software artifact of OGL+Ginkgo,
not a hardware/driver limitation of B70 Pro Level Zero. Different problem,
different fix path.
