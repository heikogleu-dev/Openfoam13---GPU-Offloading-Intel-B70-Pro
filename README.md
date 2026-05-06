# Intel Arc Pro B70 Pro + OpenFOAM CFD — Pioneer Documentation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-Foundation%2013-green)](https://openfoam.org)
[![Ginkgo](https://img.shields.io/badge/Ginkgo-1.10%20SYCL-orange)](https://github.com/ginkgo-project/ginkgo)
[![Intel Arc Pro B70](https://img.shields.io/badge/Intel%20Arc%20Pro-B70%20Pro%2032GB-blue)](https://www.intel.com/content/www/us/en/products/sku/245797)
[![GPU CFD Status](https://img.shields.io/badge/GPU%20CFD%20Status-Software%20Not%20Ready-red)]()

> **TL;DR: The hardware is excellent. The plumbing around it is not.**
>
> GPU FP64: 1374 GFLOPS (96% of spec) ✅
> VRAM: 530 GB/s sustained (87% of spec) ✅
> Kernel-launch: 5.6 µs sync (CUDA-par) ✅
> **GPU active during CFD: only 34% of wall clock** ❌
> **GPU idle: 66% — waiting for MPI Allreduce + forced host-buffer copies** ❌

First public documentation of OpenFOAM GPU acceleration (OGL/Ginkgo/SYCL)
on Intel Arc Pro B70 Pro (Battlemage) for automotive CFD.
Honest results, real bugs found, practical fixes documented.

---

## Hardware Performance — B70 Pro is Capable

| Metric | Measured | Spec | Efficiency |
|---|---|---|---|
| **FP64 Compute** | **1374 GFLOPS** | 1430 GFLOPS | **96%** ✅ |
| FP32 Compute | 12364 GFLOPS | 22940 GFLOPS | 54% |
| **VRAM Bandwidth (sustained)** | **530 GB/s** | 608 GB/s | **87%** ✅ |
| **Kernel-launch latency (sync)** | **5.6 µs** | ~5 µs (CUDA) | **on par** ✅ |
| PCIe H2D Transfer | 15.5 GB/s | ~50 GB/s | 31% (SYCL driver limit) |
| **GPU active during CFD** | **34 % of wall clock** | — | direct xe gtidle measurement |
| GPU clock when active | 2528 MHz avg / 2800 max | 2800 MHz | 90 % of boost |
| Power (CFD load, when active) | 181 W | 275 W max | 66% TDP |

**The B70 Pro delivers excellent raw performance for compute workloads.**
FP64 and VRAM bandwidth — the two metrics that matter most for CFD —
are both above 87% efficiency. This is better than many data center GPUs.

---

## Why GPU Acceleration Doesn't Win Yet — Software, Not Hardware

| Root Cause | Impact |
|---|---|
| ~~Level Zero kernel launch latency: ~100 µs~~ → **CORRECTED: 5.6 µs (fine)** | none — see [findings/14](findings/14_kernel_launch_latency_revision.md) |
| **No GPU-aware MPI for xe driver** | `forceHostBuffer=true` mandatory → PCIe round-trip per halo exchange |
| **GPU idle 66% of wall clock** | MPI Allreduce + PCIe copies dominate over compute — see [profiling/bottleneck_analysis.md](profiling/bottleneck_analysis.md) |
| **BJ(1) hits maxIter=200 cap** | every CG iteration triggers another Allreduce |
| **No working SYCL preconditioner for distributed CFD** | Multigrid OOM, IC NotImplemented, ICT DEVICE_LOST |

GPU active fraction during CFD measured directly via xe driver
`gtidle/idle_residency_ms`: only **34 % of wall clock**. When active, the
GPU runs at full boost (avg 2528 MHz). The other **66 % is idle** —
waiting for MPI Allreduce barriers and PCIe host-buffer copies.

> "It's not the GPU compute. It's the wait between compute calls —
> 66 % of wall clock spent on MPI synchronisation and PCIe host-buffer copies."

---

## CFD Benchmark Results

**Case:** MR2 SW20 vehicle aerodynamics, 34M cells, simpleFoam k-ω SST
**Metric:** Steady-state s/step (mean of Time=8,9,10)

| Configuration | s/Step | Iterations | Verdict |
|---|---|---|---|
| **CPU GAMG, 16 cores (P+E)** | **35.7** | 5–10 ✅ | **Winner — fair comparison** |
| GPU OGL BJ, np=16 (reduced quality) | 30.6 | 81 (capped) | Unfair — fewer correctors |
| GPU OGL BJ, np=8 (fair, same settings) | 53.5 | 200 (never converges) | 1.5× slower than CPU |
| GPU OGL BJ, np=8 (early test) | 52.0 | 200 (never converges) | |
| CPU GAMG, 8 P-cores only | 43.3 | 5–10 ✅ | |

**CPU GAMG with algebraic multigrid converges in 5–10 iterations.**
**GPU BJ (point-Jacobi) never converges — always hits the 200-iteration cap.**
This is an algorithmic mismatch, not a hardware limitation.

When settings are equal, **CPU wins by 1.5×**.
GPU only wins when the solver quality is artificially reduced — which
would make CPU GAMG equally or more faster at those settings.

---

## What Would Make GPU Win

Concrete s/step estimates derived from the bottleneck breakdown
(see [profiling/bottleneck_analysis.md](profiling/bottleneck_analysis.md)):

| Improvement | Estimated effect on s/step | Timeline |
|---|---|---|
| **GPU-aware MPI for xe driver** | **−10 to −20 s/step** (eliminate PCIe round-trip + reduce Allreduce serialisation) | 2–3 years |
| **Working SYCL Multigrid** (or any strong preconditioner) | **−20 to −30 s/step** (5–10× fewer iterations) | Ginkgo 2.0+ migration of OGL |
| **SYCL Graph batched submission** | −5 s/step (collapse 600 launches/step into a few) | 1–2 years |
| All three combined | could plausibly reach **<10 s/step** = ~3.5× faster than CPU GAMG (35.7 s/step) | uncertain |

The idle-time-dominated profile (66 % wall-clock idle) means **fixing the
preconditioner has higher ROI than optimising kernels** — every iteration
saved removes both its compute share AND its share of the MPI Allreduce
wait time.

---

## System

| Component | Spec |
|---|---|
| GPU | Intel Arc Pro B70 Pro (Battlemage G31) — 32 GB GDDR6 ECC |
| CPU | Intel Core Ultra 9 285K (8P+16E Cores) |
| RAM | 96 GB DDR5-6800 |
| Mainboard | ASRock Z890I Nova WiFi |
| OS | Ubuntu 26.04 LTS, Kernel 7.0.0-15 |

## Software Stack

| Component | Version |
|---|---|
| OpenFOAM | Foundation 13 |
| OGL (OpenFOAM-Ginkgo Layer) | dev branch, rebuilt with `GINKGO_JACOBI_FULL_OPTIMIZATIONS=ON` + `-fp-model=precise` |
| Ginkgo | 1.10 (OGL-internal) |
| Intel oneAPI | 2026.0.0 |
| Intel Compute Runtime | **26.05.37020.3-1** (pinned/held — see [findings/13](findings/13_stack_update_zeinit_race.md): 26.14 breaks multi-rank) |
| Intel Graphics Compiler | **2.32.7** (Intel rolling, kept after rollback) |

---

## Bugs Found & Documented (All New)

| # | Bug | Impact |
|---|---|---|
| [01](findings/01_pcie_reporting_bug.md) | lspci reports PCIe 1.0×1 (xe SR-IOV PF bug) | Diagnostic confusion |
| [02](findings/02_bj_blocksize_int_underflow.md) | `size_t` underflow in `dpcpp::jacobi::find_blocks` for BJ>1 | Confirmed via direct VRAM measurement (peak 8.4/27.9 GB) |
| [03](findings/03_preconditioner_subdict_syntax.md) | OGL preconditioner sub-dict syntax undocumented | Options silently ignored |
| [04](findings/04_sycl_device_selector.md) | `ONEAPI_DEVICE_SELECTOR=level_zero:0` (not `gpu:0`) | Crash on startup |
| [05](findings/05_sycl_preconditioner_status.md) | IC `NotImplemented` + ICT/ILU SYCL apply gaps (`lower_trs` absent) | No ILU family on SYCL — see KIT clarification |
| [08](findings/08_multigrid_device_lost.md) | Multigrid OOM in PGM coarsening + divergence | No GPU multigrid |
| [09](findings/09_pcie_nvtop_confirmation.md) | nvtop confirms PCIe Gen5×16 (lspci wrong) | Diagnostic only |
| [10](findings/10_ginkgo2_api_breaks.md) | Ginkgo 2.0 API breaks OGL (3 locations) | Cannot upgrade yet |
| [12](findings/12_cpu_tuning_no_effect.md) | CPU tuning has zero effect (DDR5 bandwidth-bound) | 35.7 s/step is hard CPU limit |
| [13](findings/13_stack_update_zeinit_race.md) | CR 26.14 incompatible with multi-rank OGL | Stack pinned at 26.05 |
| [14](findings/14_kernel_launch_latency_revision.md) | Kernel-launch latency 5.6 µs (NOT ~100 µs) | Bottleneck story revised |
| [15](findings/15_scaling_for_spd_preconditioners.md) | `scaling -1.0` correct for SPD but cannot bridge SYCL impl gaps | OGL doc was right, blocked by separate bugs |
| [16](findings/16_splitcomm_test.md) | `splitComm=false` has no effect | Eliminated as tuning knob |
| [17](findings/17_hybrid_solver_test.md) | Hybrid (CPU GAMG `p` + GPU `U/k/ω`) shows no net gain | Tight coupling needs tighter integration |

---

## Repository Structure

```
├── README.md              — This file
├── hardware.md            — Full hardware specs and measured performance
├── conclusions.md         — Honest CPU-vs-GPU verdict
├── references.md          — Cross-references to upstream papers + projects
├── setup/
│   ├── install_stack.md   — OGL/Ginkgo/oneAPI installation + required patches
│   └── bios_settings.md   — BIOS optimization for compute workloads
├── benchmarks/
│   ├── results.md         — All CFD benchmark results
│   └── hardware_results.json — Machine-readable hardware metrics
├── profiling/             — Bottleneck + VRAM analysis
│   ├── bottleneck_analysis.md — Where do the 53 s/step actually go?
│   └── vram_analysis.md       — Direct xe-debugfs VRAM measurement
├── findings/              — 14 bug findings (01–05, 08–10, 12–17)
├── configs/               — Working fvSolution configurations
└── logs/                  — Raw diagnostic logs for upstream debugging
    └── vram-traces/       — CSVs + mpirun logs from the VRAM measurement
```

---

## Status: May 2026

**Current recommendation:** Use CPU GAMG for production CFD on this hardware.
The 32 GB ECC VRAM is excellent for LLM inference workloads in the meantime.

**Watch for:** OGL Ginkgo 2.0 migration, SYCL triangular-solve kernel
implementation, GPU-aware MPI for the `xe` driver. Re-evaluate when at
least one of these materialises.

*Pioneer documentation independently maintained.*
*Full reproducibility intended for the next Battlemage CFD pioneer.*

---

## When to Re-evaluate GPU Offloading

The current limitation is not hardware — it is the software stack.
Direct VRAM measurement (see [`profiling/vram_analysis.md`](profiling/vram_analysis.md))
shows BJ(1) peaks at 9.4 GB of 27.9 GB available — **19.9 GB headroom**
exists for stronger preconditioners. The blockers are integer-arithmetic
bugs and missing kernels in the SYCL backend, not memory pressure.

Re-test when **at least one** of these conditions is met:

| Condition | Expected Gain | Status |
|---|---|---|
| OGL migrates to Ginkgo 2.0 | BJ maxBlockSize > 1, ParIC / Multigrid available | Waiting on KIT — see [findings/10](findings/10_ginkgo2_api_breaks.md) |
| Ginkgo SYCL IC / ILU production-ready | Real preconditioner: 20–50 iter vs current 200 (cap) | Unclear — see [findings/05](findings/05_sycl_preconditioner_status.md) |
| GPU-aware MPI for `xe` driver | Eliminates `forceHostBuffer`: ~−13 % per step | 1–2 years |
| Compute Runtime ≥ 26.14 stable for multi-rank | Removes the 26.05 pinning — see [findings/13](findings/13_stack_update_zeinit_race.md) | Filed upstream |

**Minimum viable re-test:** Ginkgo 2.0 + OGL migration complete.
**Optimistic timeline:** 12–18 months (mid-2027).
**Watch:** [hpsim/OGL](https://github.com/hpsim/OGL) and
[ginkgo-project/ginkgo releases](https://github.com/ginkgo-project/ginkgo/releases)

---

## How to Cite

If this documentation helped your research or work, please cite:

```bibtex
@misc{gleu2026battlemage,
  author = {Gleu, Heiko},
  title  = {Intel Arc Pro B70 Pro + OpenFOAM CFD: Pioneer Documentation
             of GPU Acceleration via OGL/Ginkgo/SYCL on Battlemage},
  year   = {2026},
  url    = {https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro}
}
```

## Related Projects

- [hpsim/OGL](https://github.com/hpsim/OGL) — OpenFOAM Ginkgo Layer (GPU solver plugin)
- [ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo) — Ginkgo linear algebra library
- [OpenFOAM/OpenFOAM-Intel](https://github.com/OpenFOAM/OpenFOAM-Intel) — Intel's official OpenFOAM contributions
- [Phoronix B70 Pro Linux Benchmarks](https://www.phoronix.com/review/intel-arc-pro-b70-linux) — Reference benchmarks, same hardware

## Community & Discussion

Found different results on your hardware? Have a fix?
→ [Open an Issue](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/issues)

General GPU-accelerated CFD discussion:
→ [CFD-Online OpenFOAM Forum](https://www.cfd-online.com/Forums/openfoam/)
→ [Reddit r/CFD](https://reddit.com/r/CFD)
→ [Reddit r/IntelArc](https://reddit.com/r/IntelArc)
