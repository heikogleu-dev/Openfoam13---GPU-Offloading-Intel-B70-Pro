# Honest Conclusions

> ## ⏱ Update — June 2026 (supersedes the May verdict below)
>
> Two months of follow-up work changed the picture substantially. The
> May conclusions (kept below as the historical record) were correct for
> their stack but several of their blockers are now resolved. Current
> honest state:
>
> ### The SYCL preconditioner bugs are FIXED in Ginkgo 2.0
> The May verdict's central claim — "no viable GPU preconditioner exists"
> — no longer holds. A standalone Ginkgo 2.0 SYCL sweep
> ([findings/26](findings/26_ginkgo_2.0_standalone_sweep.md)) confirms
> **four** previously-blocking bugs are fixed: `find_blocks` underflow
> (BJ>1), `add_candidates` SIGABRT (ICT), `lower_trs` NotImplemented
> (ILU, via [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023)),
> and Multigrid PGM. BJ(1/2/4/8/16), ILU, ISAI and Multigrid all run
> single-process up to 36M rows. OGL was migrated to Ginkgo 2.0 with two
> small patches ([findings/24](findings/24_pr168_patched_ilu_first_test.md)).
>
> ### The new blocker is an Intel Compute Runtime regression, not Ginkgo
> CR 26.14–26.18 abort during **multi-process** Level-Zero `zeInit`
> (`gmm_helper/resource_info.cpp:15`) whenever ≥2 ranks share the GPU —
> the normal mode for MPI CFD. Pure-Level-Zero minimal reproducer +
> full analysis in [findings/29](findings/29_cr_26.18_root_cause_pure_l0_multiprocess_abort.md);
> upstream as [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922)
> (open, no fix, persists through CR 26.18 + kernel 7.0.0-22).
>
> ### Workaround that restores multi-rank OGL: CR 26.05 LD-switch
> User-side `LD_LIBRARY_PATH` to an extracted CR 26.05 `libze_intel_gpu`,
> no sudo, no system change ([findings/27](findings/27_cr2605_ld_switch_workaround.md),
> `scripts/cr2605-shell.sh`). On a freshly-booted GPU this ran BJ(1)
> multi-rank at 34M cells again (~94 s/step).
>
> ### The performance question is now ANSWERED (clean-boot run, Finding 30)
> A freshly-booted GPU + the CR 26.05 LD-switch finally produced
> deterministic multi-rank numbers on the 34M case
> ([findings/30](findings/30_post_recovery_clean_multirank_perf.md)):
> - **BJ(1): ~51.5 s/step** steady-state (the earlier "~94 s/step" was the
>   setup-inclusive *first* step). Every pressure solve hits the
>   201-iteration cap — BJ(1) never converges. → **~1.44× slower than CPU
>   GAMG (35.7 s/step).**
> - **ILU: VRAM OOM → DEVICE_LOST** at peak 31.5 GB, in the `ParIlu`
>   factorization's `Csr::convert_to(Coo)` — does **not fit** at 34M cells
>   on the 32 GB B70 with the OGL distributed overhead. (GPU recovered on
>   its own, no reboot.)
> - **BJ(2–16): still the `find_blocks` size_t underflow** in the OGL
>   distributed path (clean `AllocationError`, GPU unharmed).
>
> So no *strong* GPU preconditioner currently works on this mesh at this
> VRAM: the one that runs (BJ1) is too weak, the ones strong enough to win
> either don't fit (ILU/MG) or hit the distributed `find_blocks` bug
> (BJ>1). The B70 ran 8-way multi-rank GPU solves cleanly — this is a
> **software-stack** limit, not hardware.
>
> ### The conclusive test (half-resolution 7.1M mesh) — VRAM was a red herring
> To remove VRAM from the equation we remeshed the case at half
> resolution (7.1M cells, parallel snappyHexMesh on 8 cores) and ran CPU
> GAMG vs GPU ILU on the **same mesh / decomposition / initial condition**
> ([findings/30, Addendum 2](findings/30_post_recovery_clean_multirank_perf.md)):
>
> | per step | GAMG (CPU np=8) | ILU (GPU B70 np=8) |
> |---|---|---|
> | wall-clock | **~7.7 s** | **~22–24 s** |
> | pressure CG iters | **3–5** | **160–201** |
> | peak VRAM | host | 10.7 GB (fits easily) |
>
> ILU on the GPU is **~3× slower than CPU GAMG even where VRAM is a
> non-issue**, because it needs ~40× more CG iterations (ILU is a local
> preconditioner; GAMG is algebraic multigrid with ~O(N) convergence).
>
> ### Update — Ginkgo Multigrid DOES work (the path forward exists)
> Testing every available preconditioner on the 7.1M mesh (np=8) changed
> the conclusion: **Ginkgo's algebraic Multigrid runs, fits (10.4 GB), and
> converges in 55–101 iterations at ~14 s/step — only ~1.8× slower than CPU
> GAMG (7.7 s)**, vs ILU's ~3× (160–201 iters) and BJ/ISAI never
> converging. And this is the *untuned* default (V-cycle, Jacobi smoother).
>
> | preconditioner (7.1M, np=8) | iters/solve | s/step | VRAM |
> |---|---|---|---|
> | GAMG (CPU) | 3–5 | 7.7 s | host |
> | **Multigrid (Ginkgo GPU)** | **55–101** | **~14 s** | 10.4 GB |
> | ILU (GPU) | 160–201 | ~23 s | 11.0 GB |
> | ISAI / BJ(1) (GPU) | 201 (cap) | — | — |
> | ICT / BJ(>1) (GPU) | crash | — | — |
>
> ### Net
> "Hardware great, software not *yet* ready" — but the path is now clear
> and matches the field. BJ/ILU are one-level preconditioners that lose to
> multigrid (textbook; AmgX/Hypre prove GPU-AMG is the answer). On this
> stack the GPU-AMG exists — **Ginkgo Multigrid** — and already lands
> within ~1.8× of CPU GAMG untuned at 7.1M. The remaining work is concrete
> and bounded: (1) **tune** Ginkgo Multigrid (W-cycle, stronger smoother,
> more levels) to close the ~1.8× gap, and (2) make it **VRAM-viable at
> production mesh sizes** (~1027 bytes/row today). Until then: CPU GAMG for
> production; but a real B70 GPU-CFD win is now a tuning problem, not a
> dead end.

---

## Does GPU Acceleration Help for This Case? (May 2026 — historical)

**Short answer: No, not with the current software stack (May 2026).**

At equal solver settings (nNonOrth=2, maxIter=200, all equations on the
same hardware path):
- **CPU GAMG np=16: 35.7 s/step**
- **GPU OGL BJ np=16: ~50 s/step**

GPU only "wins" (30.6 s/step) with reduced quality settings (nNonOrth=1,
maxIter=80) that would also speed up CPU GAMG to an estimated ~24 s/step.

## Why GPU Loses on This Workload

| Factor | Impact |
|---|---|
| ~~Level Zero kernel launch latency~~ — measured at **5.6 µs**, on par with CUDA (see [findings/14](findings/14_kernel_launch_latency_revision.md)) | NOT a bottleneck — earlier "100 µs" claim was wrong |
| No GPU-aware MPI: forced D2H→MPI→H2D round-trip per AllToAll | 5-10× extra communication cost |
| No GPU-side Multigrid: BJ vs GAMG = O(√N) vs O(log N) iterations | algorithmic disadvantage |
| SYCL preconditioner gaps: IC/ILU/IRILU/Hybrid not implemented or broken | Limited tuning options |
| Block-Jacobi maxBlockSize=1 only stable choice | Weak preconditioning |

## What the B70 Pro IS Good At

1. **LLM Inference:** 32 GB ECC VRAM enables serious local AI work
   (Gemma 27B + Qwen2.5-Coder 32B simultaneously loaded)
2. **ParaView Visualization:** ANARI helide backend (CPU Embree currently;
   GPU helide_gpu would need SDL3+glslang)
3. **Future CFD potential:** SYCL Graph + GPU-aware MPI could 5-10× improve
   the CFD path — but those are 12-24 months out

## Realistic Timeline for GPU CFD Improvement

| Timeframe | Expected Change |
|---|---|
| Now | Use CPU GAMG, GPU for LLMs |
| 6-12 months | Level Zero command-list/Graph optimizations land in icpx |
| 12-18 months | SYCL Graph integrated in Ginkgo + OGL |
| 18-24 months | GPU-aware OpenMPI/UCX with Level Zero support |
| 24-36 months | Production-grade GPU CFD on Battlemage class hardware |

## Bugs Reported / Documented

In this repo:
1. [lspci PCIe speed reporting bug](findings/01_pcie_reporting_bug.md) (xe driver, SR-IOV PF mode)
2. [Ginkgo SYCL Block-Jacobi `find_blocks` underflow with maxBlockSize > 1](findings/02_bj_blocksize_int_underflow.md) (fixed in Ginkgo 2.0, see June update)
3. [OGL preconditioner sub-dict syntax requirement](findings/03_preconditioner_subdict_syntax.md) (undocumented)
4. [ONEAPI_DEVICE_SELECTOR syntax pitfall](findings/04_sycl_device_selector.md)
5. [SYCL preconditioner support matrix](findings/05_sycl_preconditioner_status.md) (only BJ at maxBS=1 stable)

Plus:
- xe driver in OpenFOAM apt-paraview pulls in broken `pvserver --version` (hangs)
- Hybrid matrix format claimed in OGL README, not implemented in distributed mode
- `ICT` preconditioner causes GPU `DEVICE_LOST` during ParICT generate

## Recommendation

For production CFD on this hardware in May 2026:
- **p:** CPU GAMG, np=16 (all P+E cores)
- **U/k/ω:** CPU PBiCGStab/DILU
- **GPU:** Reserved for LLM inference (vLLM/llama.cpp with SYCL backend)

Revisit GPU-CFD acceleration in 12–18 months when SYCL Graph lands in OGL
or when AMD/NVIDIA-style GPU-aware MPI becomes available for Intel Xe Consumer.

## What Worked Despite Everything

This is still pioneering work — and the stack DOES build and run. We have:
- Working OGL/Ginkgo/SYCL build on Battlemage with patches
- Verified bandwidth profile (508 GB/s VRAM, 15 GB/s PCIe)
- Reproducible benchmark methodology (Time=8-10 mean)
- Identified specific bugs upstream can fix
- Clear path forward (Ginkgo 2.0, SYCL Graph, GPU-aware MPI)

That's a useful foundation for the next iteration when the software catches
up to the hardware.

## Honest Self-Assessment

We spent significant effort here. The result is a **negative result** in
the sense that GPU doesn't beat CPU for THIS case TODAY. That's still
valuable — it's an honest data point that other Battlemage CFD users can
build on instead of repeating our 30+ failed runs.

The hardware is great. The software is not yet ready. ¯\_(ツ)_/¯

## Addendum: All Preconditioner Tests Completed (May 2026)

After exhaustive testing, the situation is clear:

**No viable GPU preconditioner exists in Ginkgo 1.10 SYCL for this mesh.**

The only working option (BJ maxBlockSize=1) is mathematically equivalent to
diagonal scaling — far too weak for a 34M-cell CFD pressure system.
It never converges; the solver always hits maxIter=200 cap.

**Path forward: OGL rebuild with Ginkgo 2.0**

Ginkgo 2.0 includes:
- Improved SYCL BJ generate (potential fix for maxBlockSize>1 OOM)
- Better distributed multigrid
- More complete SYCL preconditioner support

This rebuild is the only remaining avenue for meaningful GPU acceleration.

> **Outcome (June 2026):** This path was taken and the Ginkgo-side bugs
> are indeed fixed — see the June update at the top of this file. The
> remaining blocker moved to the Intel Compute Runtime (multi-process
> `zeInit` abort), with a working CR 26.05 LD-switch workaround. The
> "does it actually beat CPU GAMG with a strong preconditioner"
> question is now testable for the first time but not yet answered.
