# Finding 30 — Clean-boot multi-rank validation + the definitive preconditioner verdict (34M cells, B70)

**Date:** 2026-06-17 (after the reboot that cleared the post-cascade
degraded driver state from Findings 27–29)
**Stack:** OF Foundation 13, OGL PR #168 + 2 patches, Ginkgo 2.0.0,
oneAPI 2026.0, CR 26.05 via LD-switch (Finding 27), Intel Arc Pro B70
(BMG-G31, 32 GB), 8 MPI ranks single-GPU.

This finding closes the one question left open in `conclusions.md` (June
update) and in the greole status doc: **does a *strong* GPU
preconditioner beat CPU GAMG (35.7 s/step) on the 34M-cell case?** The
earlier attempts never produced a clean number — they crashed pre-solve
on CR 26.18 or hit a degraded GPU. With a freshly-booted GPU + the CR
26.05 LD-switch, all tests ran deterministically.

## Method: device-lost-safe ramp (cheapest GPU op first)

To avoid burning the GPU on a single bad run, the tests were ordered from
least to most GPU-stressing, verifying health at each step:

| Stage | Test | Result |
|---|---|---|
| A1 | `diag-l0` single-process L0 alloc (1 KiB → 100 MiB) | ✅ PASS — GPU healthy post-boot |
| A2 | `diag-ginkgo` single-process Ginkgo 2.0 CG (executor/SpMV/BJ) | ✅ PASS |
| B1 | `diag-mpi-l0` np=2 — two processes share GPU via `zeInit` | ✅ PASS (105 ms) — CR 26.05 fixes the CR 26.18 abort |
| B2 | `diag-mpi-ginkgo-solve` np=2, 1M rows/rank, 50 CG iter | ✅ PASS (~116 ms) |
| C  | `diag-mpi-ginkgo-solve` np=8, 4M rows/rank (~32M aggregate), 100 CG iter | ✅ PASS (~1.27 s) |

The full multi-rank GPU pipeline is healthy from 1 → 8 ranks after a
clean boot with the CR 26.05 LD-switch. Per-CG-iteration baseline at 4M
rows/rank: ~12.7 ms.

## Stage D — the real 34M case, three preconditioners

Case: `Testcase-GPU`, 34,088,573 cells, scotch decomposition np=8,
`incompressibleFluid`, SIMPLE `nNonOrthogonalCorrectors 2` (→ 3 pressure
solves/step), `GKOCG` p-solver, `maxIter 200`. CPU GAMG reference (from
`conclusions.md`): **35.7 s/step (np=16)**.

### D1 — BJ(maxBlockSize=1): runs, but too weak

| Step | ExecutionTime | Δ/step |
|---|---|---|
| 1 | 93.9 s | (incl. OGL/Ginkgo setup) |
| 2 | 145.9 s | **52.0 s** |
| 3 | 197.1 s | **51.2 s** |

- **Steady-state ≈ 51.5 s/step.** *Correction:* the "~94 s/step" figure
  in the June `conclusions.md` update was the **setup-inclusive first
  step** — the true per-step cost is ~51.5 s, consistent with the May
  "~50 s/step" measurement.
- Every pressure solve hits the **201-iteration cap** (initial residual
  → final residual barely reduced, e.g. 1 → 0.874 on the first solve).
  BJ(1) ≈ diagonal scaling — far too weak for a 34M pressure system, as
  long established.
- **Verdict:** GPU BJ(1) 51.5 s/step vs CPU GAMG 35.7 s/step → **GPU
  ~1.44× slower**, because it never converges (caps at 201 iter) where
  GAMG converges in a handful of V-cycles.

### D2 — ILU: VRAM OOM → DEVICE_LOST at the factorization

ILU now reaches `apply` on SYCL (Ginkgo PR #2023 fixed `lower_trs`), so
the test is finally meaningful on Ginkgo 2.0. It crashed on the **first
pressure solve**, during the factorization:

```
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
stack: gko::matrix::Csr<d,i>::convert_to(Coo)
       → gko::Executor::copy_from → gko::DpcppExecutor::raw_copy_to
       (inside gko::factorization::ParIlu::generate_l_u)
```

- **Peak VRAM (fdinfo sum over 8 ranks): 32,212 MiB ≈ 31.5 GB** — pinned
  against the 32 GB ceiling.
- This is the spike predicted in Finding 24: `ParIlu::generate_l_u`
  converts the per-rank `Csr` shard to `Coo`, and that copy/alloc pushes
  total VRAM past the hardware limit → the Level-Zero backend returns
  `DEVICE_LOST` mid-copy.
- **Verdict:** ILU does **not fit** at 34M cells on the 32 GB B70 with
  the current OGL distributed overhead. Would need either a smaller mesh,
  a 48/64 GB card, or an OGL/Ginkgo factorization path that avoids the
  full Csr→Coo materialization.
- **GPU recovered on its own** — a follow-up single-process `diag-l0`
  passed immediately, no reboot needed. (Contrast the older BJ(2) crash
  in the test plan that left a stuck cascade; the `intel_iommu=igfx_off`
  GRUB fix + fresh driver state now clean up the DEVICE_LOST.)

### D3 — BJ(maxBlockSize=8): the `find_blocks` underflow, confirmed on Ginkgo 2.0

```
terminate called after throwing an instance of 'gko::AllocationError'
  what():  .../dpcpp/base/executor.dp.cpp:104: DPC++: failed to allocate
           memory block of 18446744073709551615B
```

- `18446744073709551615 = 2^64 − 1` — the classic `size_t` underflow in
  the distributed-path block detection (Findings 02 / 26).
- This is a **clean CPU-side `AllocationError`** (a 16-EB allocation is
  refused before any GPU work) — **not** a DEVICE_LOST. GPU stayed
  healthy (`diag-l0` PASS afterwards).
- Confirms the greole-doc §3 localization: standalone `Jacobi(bs>1)`
  runs to 36M rows (Finding 26), but the OGL distributed Schwarz wrapper
  underflows for any `maxBlockSize > 1`. The trigger is the per-rank
  shard fed into `find_blocks`, not the Ginkgo kernel itself.

## The definitive verdict (closes the open question)

On the 34M-cell case at 32 GB, **no strong GPU preconditioner currently
works**:

| Preconditioner | 34M / 32 GB outcome |
|---|---|
| BJ(1) | ✅ runs, but ~51.5 s/step and never converges (201-iter cap) → 1.44× slower than CPU GAMG |
| BJ(2–16) | ❌ `find_blocks` size_t underflow in OGL distributed path |
| ILU | ❌ VRAM OOM → DEVICE_LOST at `Csr::convert_to(Coo)` (peak 31.5 GB) |
| Multigrid | ❌ (predicted) ~35 GB > 32 GB ceiling — not attempted, see test plan |

So the honest answer to *"does the GPU beat CPU GAMG with a strong
preconditioner?"* is: **not on this mesh at this VRAM, with the current
OGL + Ginkgo 2.0 stack.** The only preconditioner that runs (BJ1) is too
weak to converge; the ones strong enough to plausibly win either don't
fit (ILU/MG) or hit the distributed `find_blocks` bug (BJ>1).

This is not a hardware limit — the B70 ran 8-way multi-rank GPU solves
cleanly (Stage C) and the per-iteration throughput is good. It is a
**software-stack** limit: (1) the OGL distributed `find_blocks` underflow
blocks the cheap strong-ish option (block-Jacobi), and (2) the ILU
factorization's Csr→Coo materialization blows the 32 GB budget at this
mesh size.

## Where this moves the upstream story

- **greole / OGL #170:** the §5 "pending" row is now answered — BJ(1)
  baseline = 51.5 s/step (not 94), ILU OOMs at the factorization on 32 GB
  / 34M, BJ>1 still underflows. The single highest-value OGL fix remains
  the distributed `find_blocks` path; a VRAM-leaner ILU generate would
  unblock the strong-preconditioner case on 32 GB cards.
- **Practical guidance unchanged:** for production CFD on this hardware
  today, CPU GAMG. The GPU door opens with either a larger-VRAM card
  (ILU would then fit) or the `find_blocks` distributed fix (BJ>1).

## Addendum (2026-06-17, evening) — smaller mesh (30.5M) ILU attempt

To test whether ILU fits on a *smaller* mesh, we ran it on `Oem30`
(30,481,275 cells, ~11% smaller than the 34M case), decomposed to 8
ranks (~3.82M/rank), restarting from a near-converged CPU solution
(t=2000). Note: after the GRUB change that removed the iGPU-PRIME
passthrough, the **desktop now runs on the B70** (monitor attached to it),
consuming ~1.15 GB — so less VRAM headroom than the earlier 34M run had.

**Result — the first ILU pressure solve actually completed:**

```
ILUsyclGKOCG:  Solving for p, Initial residual = 0.03746, Final residual
               = 3.607e-05, No Iterations 181
```

- **ILU converged** — 0.0375 → 3.6e-05 (relTol 1e-3) in **181 CG
  iterations**. Contrast BJ(1), which caps at 201 iterations without
  converging. This is the **first strong-preconditioner data point on the
  B70**: ILU is genuinely stronger than block-Jacobi here.
- It then hit `UR_RESULT_ERROR_DEVICE_LOST` on a **subsequent** solve.
  Peak foamRun VRAM **31,565 MiB** (fdinfo, 8 ranks) + ~1.15 GB desktop
  ≈ **32.7 GB total** → over the ceiling.
- **The peak barely scaled down** from the 34M run (32,212 → 31,565 MiB
  foamRun): the `Csr::convert_to(Coo)` materialization in `ParIlu` is
  largely mesh-size-insensitive at the spike, so an 11% smaller mesh did
  **not** buy enough headroom — and the desktop-on-B70 ate what little it
  did. GPU self-recovered (`diag-l0` PASS), no reboot.
- Multigrid not attempted — predicted ~35 GB > 32 GB; a 3rd OOM gamble
  was not worth it.

**Two takeaways:**
1. **VRAM:** ILU needs noticeably **<30.5M cells** on a 32 GB B70 (or a
   larger card, or the desktop off the B70). 30.5M is still over the edge.
2. **Convergence quality:** even where it ran, ILU needed **181 CG
   iterations** to drop 3 orders from an already-low restart residual —
   far more than GAMG's handful of V-cycles. So even with enough VRAM,
   ILU-preconditioned CG as configured (default `ParIlu`, no fill) would
   be unlikely to beat CPU GAMG on this pressure system without a stronger
   factorization (ILU(k)/more sweeps). The strong-preconditioner *win* is
   still not demonstrated — but for the first time the strong
   preconditioner at least *ran and converged* on the B70.

Artifact: `Oem30/log.oem30-ILU` →
`logs/post-recovery-2026-06-17/oem30-ILU.log.gz`.

## Addendum 2 (2026-06-17, late) — half-resolution mesh: the conclusive ILU-vs-GAMG comparison

Finally, a mesh small enough that ILU runs to completion: we copied the
original `Testcase`, halved the background blockMesh (120×60×40 →
60×30×20, i.e. 2× coarser in every direction), and re-ran the full
snappyHexMesh pipeline (castellate + snap + layers) **in parallel on 8
cores**. Result: a clean **7,118,582-cell** mesh (`checkMesh`: Mesh OK,
max non-orthogonality 73.5°, 2 severe faces). Decomposed to 8, started
from uniform initial fields (t=0), identical solver tolerances
(`tolerance 1e-6`, `relTol 0.01`). We ran **CPU GAMG and GPU ILU on the
exact same mesh / decomposition / initial condition** — the only
difference is the pressure solver. (U/k/ω are CPU PBiCGStab/DILU in both.)

| Metric (steady-state) | **GAMG (CPU, np=8)** | **ILU (GPU B70, np=8)** |
|---|---|---|
| Wall-clock per step | **~7.7 s** | **~22–24 s** |
| Pressure CG iterations / solve | **3–5** | **160–201** (often hits the 200 cap) |
| Peak VRAM | n/a (host) | **10.7 GB** — fits with ~20 GB to spare |
| Completes the run | ✅ | ✅ (5 steps clean, no DEVICE_LOST) |

**ILU on the GPU is ~3× slower than GAMG on the CPU — at a mesh size
where VRAM is a non-issue.** This is the conclusive result the whole
investigation was after:

- **VRAM was never the fundamental blocker.** At 7.1M, ILU peaks at 10.7
  GB — trivially fits. The 34M/30.5M OOMs were a *symptom* of pushing
  ILU to mesh sizes the 32 GB card can't hold, but even when it fits
  comfortably, ILU loses.
- **The real gap is the preconditioner's convergence rate.** ILU needs
  ~40× more CG iterations than GAMG (160–201 vs 3–5). ILU is a *local*
  preconditioner; GAMG is algebraic multigrid with near-optimal O(N)
  convergence on elliptic pressure systems. No amount of GPU
  per-iteration speed closes a 40× iteration gap when the CPU baseline
  is only 3× faster in wall-clock.
- **What would actually win:** a competitive **GPU-side algebraic
  multigrid** as the pressure preconditioner. Ginkgo has Multigrid/PGM,
  but it OOMs at these mesh sizes (Finding 26: ~1027 bytes/row) and isn't
  effective through OGL's distributed path. Until GPU AMG is available
  and VRAM-viable, block-Jacobi (too weak) and ILU (too slow to
  converge) are the only options — and neither beats CPU GAMG.

So across three mesh sizes the verdict is consistent and now *complete*:
BJ(1) runs but never converges; BJ(>1) hits the `find_blocks` underflow;
ILU converges but needs ~40× GAMG's iterations (and OOMs above ~25–30M);
Multigrid doesn't fit. **GPU pressure-solve does not beat CPU GAMG on
this class of problem with today's OGL + Ginkgo preconditioners — and the
missing piece is GPU algebraic multigrid, not more VRAM or faster
kernels.**

Mesh build: `blockMesh` (36k base) → parallel `snappyHexMesh` (8 cores,
~20 min) → 7.1M. Artifacts: `Testcase-half/log.GAMG-cpu`,
`Testcase-half/log.ILU-gpu` → `logs/post-recovery-2026-06-17/`.

## Artifacts

- `Testcase-GPU/log.post-recovery/test-BJ1.log` — clean 3-step BJ(1) run
- `Testcase-GPU/log.post-recovery/test-ILU.log` — ILU DEVICE_LOST stack
- `Testcase-GPU/log.post-recovery/test-BJ8.log` — BJ(8) find_blocks underflow
- `gpu-diag/` — diag-l0 / diag-ginkgo / diag-mpi-* reproducers (Stages A–C)
- `gpu-diag/run-ilu-monitored.sh` — VRAM-peak-sampling run wrapper
