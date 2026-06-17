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

## Addendum 3 (2026-06-17) — why 160–201 iterations? Rank sweep + BJ(>1), on the 7.1M mesh

The ~160–201 ILU iterations looked suspicious, so we stress-tested two
hypotheses on the 7.1M mesh.

**Hypothesis A (refuted): additive-Schwarz fragmentation.** OGL wraps the
distributed preconditioner in `distributed::preconditioner::Schwarz`, so
ILU is applied block-locally per rank. If that were the cause, more ranks
→ more blocks → more iterations. We swept np = 2/4/8/12 (each
re-decomposed, `ranksPerGPU` matched):

| np | pressure iters (steps 1–3, 3 solves each) | s/step (steady) |
|---|---|---|
| 2 | 201,161,160 / 143,201,201 / 201,201,201 | ~32 s |
| 4 | 201,161,161 / 143,201,201 / 201,201,201 | ~26 s |
| 8 | 201,161,161 / 142,201,201 / 201,201,201 | ~23 s |
| 12 | 201,161,161 / 142,201,201 / 201,201,201 | ~23 s |

**Iteration count is essentially rank-independent** (np=2 ≈ np=12). So it
is *not* a Schwarz/decomposition artifact and not a parallel-config error.
(np=1 can't be tested: OGL requires `-parallel`, OpenFOAM refuses
`-parallel` with a single rank — `UPstream::init` aborts.) Wall-clock does
scale with ranks up to np=8, then saturates at np=12 — expected, since all
ranks share one GPU.

**What it actually is: textbook ILU(0) convergence.** The pressure CSR
matrix has no fill-in preconditioner available beyond ILU(0) (zero fill)
in the OGL/Ginkgo SYCL path. ILU(0)-preconditioned CG on a 3-D elliptic
system scales like **~N^(1/3)** iterations (mesh-diameter-bound), whereas
algebraic multigrid (GAMG) is ~O(1), mesh-independent. For 7.1M cells
N^(1/3) ≈ 192 — almost exactly the 160–201 observed. So the iteration
count is *correct expected behaviour for ILU(0)*, not a misconfiguration.

Consequence that matters: because ILU(0)-CG is ~N^(1/3) and multigrid is
~O(1), **the GPU-ILU disadvantage grows with mesh size** — at 34M, ILU
would need ~N^(1/3) ≈ 324 iterations vs GAMG's ~5. Bigger meshes make
GPU-ILU relatively *worse*, not better. A larger-VRAM card would let ILU
*run* at 34M but it would lose to GAMG by an even wider margin.

**BJ(maxBlockSize=8) and BJ(16) on 7.1M: still broken.** Both abort at the
block-Jacobi generation step (`Generate preconditioner BJ<double>
MaxBlockSize 8` → process abort) — the same `find_blocks` distributed-path
failure seen at 34M. So the BJ(>1) bug is **mesh-size-independent**; larger
blocks are not an escape hatch. GPU stayed healthy (clean abort, `diag-l0`
PASS).

**Bottom line:** the high iteration count is intrinsic ILU(0) behaviour,
not a config or parallelization bug. The only fix is a fundamentally
stronger preconditioner — GPU algebraic multigrid (mesh-independent
iterations) — which is exactly the missing piece. ILU(0) and BJ are the
wrong tool for an elliptic pressure equation no matter how they're tuned
or decomposed.

Artifacts: `Testcase-half/log.ILU-np{2,4,8,12}`, `log.BJ8`, `log.BJ16`,
`gpu-diag/ilu-rank-sweep.sh`.

## Addendum 4 (2026-06-17) — the preconditioner sweep: Multigrid works

Prompted by the right question ("it can't be that *every* preconditioner
is too weak/slow — and did you even measure VRAM?"), we ran a
VRAM-monitored sweep of all viable OGL preconditioners on the 7.1M mesh at
np=8 (`gpu-diag/precond-vram-sweep.sh`):

| Preconditioner | iters/solve | s/step | peak VRAM | result |
|---|---|---|---|---|
| GAMG (CPU) | 3–5 | 7.7 s | host | reference |
| **Multigrid (Ginkgo)** | **55–101** | **~14 s** | 10.4 GB | ✅ converges, best GPU |
| ILU | 160–201 | ~23 s | 11.0 GB | converges slowly |
| ISAI | 201 (caps) | ~15 s | 4.6 GB | too weak |
| ICT | — | — | 2.7 GB | crash before solver |
| BJ(8) | — | — | — | abort (`find_blocks`) |

**The earlier "no strong GPU preconditioner works" was wrong — Multigrid
does.** Ginkgo's algebraic multigrid converges in 55–101 iterations (real
multigrid behaviour, far below ILU's 160–201 and unlike BJ/ISAI which cap
at 201) and runs at ~14 s/step — only ~1.8× CPU GAMG, **untuned** (default
V-cycle, Jacobi smoother). This is exactly the GPU-AMG class that makes
NVIDIA AmgX win, now confirmed functional on Battlemage via Ginkgo/OGL.

Also confirmed with VRAM this time (the rank-sweep table above lacked it):
ILU peak VRAM rises with np (9.6 → 11.9 GB for np 2→12) from Schwarz
overlap; iteration count stays rank-independent.

**Revised conclusion:** the B70 GPU-CFD win is no longer a dead end but a
**tuning problem** — close the ~1.8× gap by tuning Ginkgo Multigrid
(W-cycle, stronger smoother, more levels) and make it VRAM-viable at
production mesh sizes (~1027 bytes/row). See
[knowledge/preconditioners-and-gpu-cfd.md](../knowledge/preconditioners-and-gpu-cfd.md).

Artifacts: `Testcase-half/log.{A-ILU-np*,B-ISAI-np8,B-ICT-np8,B-BJ8-np8,B-Multigrid-np8}`,
`gpu-diag/precond-vram-sweep.sh`.

## Addendum 5 (2026-06-17) — Multigrid tuning map + external cross-check

Tuned Ginkgo Multigrid (7.1M, np=8) and cross-checked against the
literature. Tuning moved it from ~14 → **~9 s/step (~1.17× CPU GAMG)**:

| MG config | iters | s/step | VRAM |
|---|---|---|---|
| default (V/Jacobi/1/Jacobi-coarse) | 55–68 | 14.0 | 11.4 GB |
| W-cycle | 14–17 | 12.1 | 10.4 GB |
| **CG coarse-solver (best)** | **12–14** | **9.0** | 11.5 GB |
| deep-coarse + CG | 9–12 | 9.2 | 10.8 GB |
| SSOR smoother | 20–25 | 29.0 | 26.5 GB |
| combo (W+SSOR+CG) | 5–7 | **127** | 26.6 GB |

External research (3 sources, see knowledge/external-references.md) explains
it precisely:
- **Fixing the coarse solver (Jacobi→CG) is the biggest lever** — matches the
  generic AMG advice; default 1× Jacobi coarse solve is weak.
- **SSOR/Gauss-Seidel are sequential on GPU → avoid** — config "combo" hits
  GAMG-like 5–7 iters but at 127 s/step + 26.6 GB. Textbook GPU-AMG uses
  Jacobi/l1-Jacobi/Chebyshev only.
- **We floor at ~13 iters, not GAMG's 3–5, because Ginkgo only has PGM**
  (unsmoothed size-2 aggregation) — no classical Ruge-Stüben / smoothed
  aggregation. AmgX/Hypre reach few iters via classical AMG (which Ginkgo lacks).
- **Fair baseline is PCG, not GAMG:** all published OGL pressure speedups
  (Olenik et al., up to 15×) are vs CPU diagonal-PCG; nobody published OGL
  beating GAMG. We're holding the GPU to a higher bar than the field.
- **AMG-resetup gotcha:** pressure coefficients change each SIMPLE iter, so the
  AMG hierarchy can't be cached like GAMG — made AmgX 2.5× slower than GAMG in
  one published benchmark. Structural headwind for any GPU-AMG.
- **Cells/GPU threshold:** need ~1–5M min, >10M ideal to beat CPU. Our 7.1M
  single-GPU run is under-fed — yet already ~1.17× GAMG.

**Next test:** a ~15–18M mesh (most that fits MG in 32 GB at ~1.6 GB/M cells)
to feed the GPU above the win threshold — the most likely route to parity/win.
Reference configs (AmgX/Hypre/Ginkgo) captured in
[knowledge/gpu-amg-reference-configs.md](../knowledge/gpu-amg-reference-configs.md).
Best B70 config set as the `Testcase-half` default.

Artifacts: `Testcase-half/log.MG-{01..08}-*`, `gpu-diag/mg-sweep.sh`.

## Artifacts

- `Testcase-GPU/log.post-recovery/test-BJ1.log` — clean 3-step BJ(1) run
- `Testcase-GPU/log.post-recovery/test-ILU.log` — ILU DEVICE_LOST stack
- `Testcase-GPU/log.post-recovery/test-BJ8.log` — BJ(8) find_blocks underflow
- `gpu-diag/` — diag-l0 / diag-ginkgo / diag-mpi-* reproducers (Stages A–C)
- `gpu-diag/run-ilu-monitored.sh` — VRAM-peak-sampling run wrapper
