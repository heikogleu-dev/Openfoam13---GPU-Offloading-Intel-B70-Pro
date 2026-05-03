# Honest Conclusions

## Does GPU Acceleration Help for This Case?

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
| Level Zero kernel launch latency: ~100 µs vs CUDA's ~5 µs | 20× overhead per Compute-Submit |
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
2. [Ginkgo 1.10 SYCL Block-Jacobi OOM with maxBlockSize > 1](findings/02_bj_maxblocksize_oom.md)
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

Ginkgo 2.0 (already installed at /opt/ginkgo) includes:
- Improved SYCL BJ generate (potential fix for maxBlockSize>1 OOM)
- Better distributed multigrid
- More complete SYCL preconditioner support

This rebuild is the only remaining avenue for meaningful GPU acceleration.
