# Are we on the right track on the Intel platform? (research synthesis 2026-06-17)

Cross-checked against GitHub, CFD forums, vendor docs, and papers. Verdict
first, evidence below. **Sources in [external-references.md](external-references.md).**

## Verdict

- **Approach: yes, aligned with the field.** CG + a matrix-built algebraic-
  multigrid preconditioner, pressure solve in FP64, is exactly what AmgX,
  Hypre/petsc4Foam and the OpenFOAM-GPU literature recommend. Native GAMG
  doesn't port (geometry-based); matrix-AMG is the right GPU path.
- **Hardware: capable, including FP64.** The B70 is *not* FP64-crippled (see
  below) — the bottleneck is the SYCL preconditioner software, not the silicon.
- **Position: bleeding edge.** As far as the public record shows, **we are the
  only ones running OGL/Ginkgo OpenFOAM offload on a consumer/Battlemage Intel
  GPU.** Official Ginkgo+OpenFOAM results are all on data-center Max/Ponte
  Vecchio; OGL's README documents only an AMD MI100 and never claims Intel
  support. Our own bug reports are the richest public source on this stack.
- **One real divergence to revisit:** multiple MPI ranks sharing one GPU (we
  do np=8) — community best practice **consolidates to one rank per GPU**.

## FP64 on the B70 is strong — not the bottleneck (corrects an earlier guess)

| GPU | FP64 | ratio | nature |
|---|---|---|---|
| Arc A770 (Alchemist) | none native | emulated only | unusable for FP64 CFD |
| **Arc Pro B70 (Battlemage, ours)** | **~1335 GFLOPS** (our clpeak, Ginkgo #2013) | **~1:8** | native, rate-limited on XVE units (not Alchemist-style emulation) |
| Data-center Max 1550 (PVC) | 52 TFLOPS | 1:1 | FP64-first |

- **We measured it ourselves:** clpeak in [Ginkgo #2013](https://github.com/ginkgo-project/ginkgo/issues/2013) → **1335 GFLOPS FP64, 530 GB/s** (93% / 87% of the internally-computed spec). `cl_khr_fp64` is exposed natively (no emulation flags needed; Ginkgo FP64 runs at full hardware rate).
- **Implication:** the ~46% compute-engine utilization is **NOT** FP64 weakness.
  Sparse CG/SpMV/multigrid are **memory-bandwidth-bound, low-arithmetic-
  intensity**; at 1335 GFLOPS FP64 / 530 GB/s the FP64 ALU ridge sits above
  SpMV intensity. The real limiters are: the Ginkgo SYCL preconditioner gaps
  (only weak ones work → many iterations), the ~30% copy-engine (host↔device,
  `forceHostBuffer`), and CPU-side assembly. **Keep FP64** — the
  weak-FP64→go-FP32 advice applies to consumer NVIDIA (1:64) and Alchemist
  (emulated), not to the B70.
- Community also keeps the pressure solve in FP64: pure FP32 fails to converge
  for turbulent LES (Brogi et al. 2022); OpenFOAM "mixed precision" = FP32
  assembly + FP64 linear algebra. So our FP64 choice matches practice.

## Where we align with the field

- **Solver/preconditioner:** CG + matrix-AMG preconditioner = the dominant
  recommended pattern (AmgX ~7× over GAMG-PCG; petsc4Foam `cg`+BoomerAMG).
- **Offloading only the pressure solve** (U/k/ω on CPU): correct — Amdahl says
  offload pays only where the linear solver dominates runtime.
- **Mixed precision plan (DP-SP, avoid FP16 on short rows):** matches the
  consensus (Carson-Higham; Ginkgo three-precision AMG "DP-SP"). FP32
  preconditioner under FP64 CG keeps full FP64 accuracy; FP16 buys ~nothing on
  a 7-nnz/row Laplacian (index traffic dominates: FP64→FP32 only ~2.33× at w=7).

## Where we diverge / open questions

- **Ranks per GPU:** we run np=8 on one GPU; AmgXWrapper and the Ginkgo
  repartitioning work **consolidate to one rank per GPU** (excess CPU ranks
  gather/scatter onto the owner rank). Concurrent multi-rank Level-Zero is also
  exactly what crashes on the B70 with CR ≥26.14 (#922). *But* our measurement
  showed more ranks = faster here (single GPU, CPU-assembly-bound). **Worth
  testing consolidation** (1 GPU-owner rank + CPU assembly) to see if it beats
  np=8 — especially at the larger 18M mesh.
- **GPU-aware MPI:** we use `forceHostBuffer` (no GPU-aware MPI) → ~30% copy
  engine + the documented 25–50% penalty. A Level-Zero-aware MPI would help.

## What's genuinely novel (our contribution)

- **Only documented consumer-Battlemage OGL/Ginkgo OpenFOAM user.** Our reports
  (OGL #170, compute-runtime #922, Ginkgo #2013/#2015/#2018) are the public
  reference for this stack.
- **`find_blocks` distributed-path underflow (BJ>1): novel, maintainer never
  confirmed.** Strongest upstream-contribution candidate. **Lead:** Ginkgo docs
  say `find_blocks` is "merely a heuristic" — passing explicit `block_pointers`
  bypasses it and *may* sidestep the underflow. Worth trying.

## Roadmap that affects us (as of 2026-06-17)

- **Classical Ruge-Stüben AMG just landed on Ginkgo `develop`** (PR #1985,
  2026-06-15) — but **CPU reference only; GPU kernels are a draft (#2034), no
  SYCL yet, unreleased.** This is the real fix for our ~13-iteration floor
  (classical AMG → few iterations like GAMG), but **not usable on the B70 yet.**
  Smoothed aggregation: still none. PMIS "on the maintainer's list."
- **Distributed multigrid already exists in Ginkgo** (v1.8.0, 2024) — "future
  work" was only at the OGL integration layer. Caveat: aggregation is local-
  per-rank; SYCL PGM coarsening had the OOM we hit.
- **Mixed-precision multigrid is exampled and works** (Ginkgo
  `mixed-multigrid-solver` instantiates `Pgm<float>`); no SYCL-specific
  mixed-precision-MG bug reported → our planned OGL DP-SP patch is feasible.
- **Ginkgo SYCL backend is DPC++/icpx-only** (#2008) and validated on data-
  center GPUs; consumer Arc is untested. Several strong preconditioners are
  `NotImplemented`/broken on SYCL in practice (our #2015).

## Honest platform verdict (record this)

Intel consumer/pro Arc is **excellent for FP32/FP16 LBM (FluidX3D — runs great,
B70 ~6750 MLUPs/s)** but **experimental/unsupported for FP64 OpenFOAM offload
via OGL/Ginkgo.** Two different risk classes. The B70 hardware (incl. its
respectable 1:8 FP64) is capable; the gap is the **Ginkgo SYCL preconditioner
software** (and, secondarily, host↔device transfer + driver maturity). We're
pioneering a path the field hasn't walked on this hardware — expect to be the
bug-finder, and the highest-leverage wins are software/upstream, not hardware.
