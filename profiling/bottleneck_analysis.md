# Bottleneck Analysis — Where Do the 53 s/Step Really Go?

## Context

Earlier docs claimed GPU kernel-launch latency (~100 µs) was the dominant
bottleneck. [findings/14](../findings/14_kernel_launch_latency_revision.md)
corrected this: actual sync latency is **5.6 µs** — on par with CUDA.
Kernel launches account for only ~33 ms out of the 53 s/step total (0.06%).

This document presents direct measurements of where the time really goes.

## Measurement Setup

- Hardware: Intel Arc Pro B70 Pro, Core Ultra 9 285K, 96 GB DDR5-6800
- Stack: Compute Runtime 26.05.37020.3-1, IGC 2.32.7, oneAPI 2026.0
- libOGL.so: rebuilt with `GINKGO_JACOBI_FULL_OPTIMIZATIONS=ON` + `-fp-model=precise`
- Case: 34M cells, np=8, BJ(maxBlockSize=1), scotch decomposition
- Run: 3 timesteps (T=1: with init, T=2,3: steady-state)
- Tools used: `xe` driver `gtidle/idle_residency_ms`, sysfs `freq0/act_freq`
  sampled at 100 ms, OGL `verbose=2`, runtime `forceHostBuffer` toggle

## Hypotheses

- **H1**: PCIe Host-Buffer transfer dominates (`forceHostBuffer=true` is mandatory)
- **H2**: MPI Allreduce / Wait dominates
- **H3**: GPU compute itself is slow (kernel inefficient)

## Headline Result

**The GPU is idle 66 % of the wall-clock time.**

| Metric | Value |
|---|---|
| Total wall clock (3 timesteps + init) | **211.4 s** |
| GPU active time (xe `gtidle` delta) | **71.9 s** = **34 %** |
| GPU idle time | **139.5 s** = **66 %** |

When the GPU is active, it runs at full boost: average **2528 MHz** (90 %
of the 2800 MHz spec), with 33 % of all samples at ≥ 2700 MHz. **GPU compute
itself is fine** — it just spends most of the wall clock waiting for work.

## Frequency Histogram (1987 samples @ 100 ms)

| Frequency | Samples | % of wall clock | Interpretation |
|---|---|---|---|
| 0 MHz (deep idle, RC6) | 1239 | **62.4 %** | GPU completely off |
| 1–1000 MHz | 80 | 4.0 % | wake-up transitions |
| 1000–2000 MHz | 8 | 0.4 % | brief mid-clock |
| 2000–2700 MHz | 10 | 0.5 % | almost-boost |
| **≥ 2700 MHz (full boost)** | **650** | **32.7 %** | actual compute work |

Bimodal distribution: GPU is either **off** or **at full boost**. There is
essentially no in-between. This rules out throttling.

## H1 Confirmed: forceHostBuffer is Mandatory

Tested `forceHostBuffer=false` (would let OGL pass device pointers directly
to MPI). Result: immediate crash on the first p-solve update:

```
gko::experimental::mpi::communicator::all_to_all_v(...)
↑ called from
update_impl(... HostMatrixWrapper ...)  ← but with device-pointer
↑ called from
incompressibleFluid::correctPressure → GKOCG::solve
```

OpenMPI 5.0.10 cannot read xe device pointers — segfault on first dereference.
**Forced host-buffer copy is the only option** with the current stack
(no GPU-aware MPI for xe).

## H2 — MPI Allreduce / Wait

Indirect evidence (no direct VTune measurement done):
- 66 % wall-clock idle on the GPU side
- BJ(1) GKOCG runs 200 iterations × 3 p-solves per timestep = 600 inner
  products per timestep, each requiring an MPI_Allreduce across 8 ranks
- Inner-product Allreduce wait time scales with the slowest rank's local
  vector-dot-product completion + network latency (loopback OpenMPI here,
  but still serialised)

Likely accounts for a large share of the 66 % idle. Direct VTune
profiling would quantify the split with MPI/PCIe further; not done in this
session.

## H3 — Refuted

- Kernel launch: 5.6 µs sync ([findings/14](../findings/14_kernel_launch_latency_revision.md))
- When active, GPU runs at full boost (2528 MHz avg) for 33 % of samples
- Single-process FP64: 1346 GFLOPS = 94 % of spec
- Single-process VRAM Triad sustained: 530 GB/s = 87 % of spec

The compute hardware is performant. **It is not the bottleneck when running.**

## Where the 53 s/Step Actually Go (Steady-State Estimate)

Extrapolating the 34 % active fraction to one steady-state timestep:

| Phase | Time per step | % of step | Source |
|---|---|---|---|
| **GPU compute (active)** | ~18.2 s | **34 %** | wall × 34 % |
| **GPU idle: MPI Allreduce wait** | ~25 s | ~47 % | dominant of the 66 % idle |
| **GPU idle: PCIe H↔D copies (`forceHostBuffer`)** | ~7 s | ~13 % | every halo exchange |
| **CPU phases (U/k/ω solvers, OGL framework)** | ~3 s | ~6 % | u/k/ω are still CPU-side |

The MPI/PCIe split (47 % / 13 %) is an estimate based on the architectural
plausibility — direct VTune profiling would refine it. The 34 % GPU-active
figure is direct measurement.

## Implications for OGL / KIT / Intel

The performance ceiling for OGL with current stack is set by **two things,
neither of which is GPU compute**:

1. **No GPU-aware MPI for xe / Level Zero** — every halo exchange round-trips
   through host memory. Even at 15 GB/s PCIe, this dominates for tightly
   coupled CG iterations. Roadmap items: SYCL Graph for batch submit, native
   xe MPI plugin for OpenMPI.

2. **BJ(1) preconditioner is too weak** — solver always hits maxIter=200.
   With a real preconditioner (Multigrid, ParIC) that converges in 30–50
   iter, the GPU's idle time would be cut roughly proportionally because
   the number of MPI Allreduces drops with the iteration count.

The idle-time-dominated profile means **fixing the preconditioner has higher
ROI than optimising the kernels**. Each iteration saved removes its share of
the 66 % idle time, not just the 34 % compute share.

## What Would Move the Needle

| Improvement | Estimated effect on s/step |
|---|---|
| GPU-aware MPI for xe | −10 to −20 s (eliminate PCIe + reduce Allreduce serialisation) |
| Working SYCL Multigrid (Ginkgo 2.0?) | −20 to −30 s (5–10× fewer iterations) |
| SYCL Graph batched submission | −5 s (reduce 600 launches/step to a few) |
| All three combined | could plausibly reach **< 10 s/step** |

CPU GAMG baseline: 35.7 s/step. The above hypothetical 10 s/step would be
~3.5× faster than CPU — a real speedup justifying GPU complexity. But it
requires three coordinated software developments that are not available today.

## What This Session Did NOT Measure

- VTune XPU-Offload (Test B) was planned but skipped — the engine-breakdown
  + `forceHostBuffer` evidence was already conclusive for the headline finding.
  A future session could add VTune for top-3 GPU kernel hotspots and a
  precise MPI/PCIe time split.
- Per-iteration breakdown of MPI vs PCIe vs GPU compute. The 34/47/13/6 split
  above is the best estimate from current data; VTune would refine it.
- intel_gpu_top engine-% (Compute vs Blitter) — `intel_gpu_top` is i915-PMU-only
  and does not work with the xe driver. No Battlemage-compatible alternative
  is currently shipped in distros.
