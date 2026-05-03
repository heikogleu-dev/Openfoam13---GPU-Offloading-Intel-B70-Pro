# Intel Arc Pro B70 Pro + OpenFOAM CFD — Pioneer Documentation

> **TL;DR: The hardware is excellent. The software stack is not ready yet.**
>
> GPU FP64: 1335 GFLOPS (93% of spec) ✅
> VRAM: 530 GB/s sustained (87% of spec) ✅
> OpenFOAM GPU acceleration: limited by Ginkgo 1.10 SYCL maturity ❌

First public documentation of OpenFOAM GPU acceleration (OGL/Ginkgo/SYCL)
on Intel Arc Pro B70 Pro (Battlemage) for automotive CFD.
Honest results, real bugs found, practical fixes documented.

---

## Hardware Performance — B70 Pro is Capable

| Metric | Measured | Spec | Efficiency |
|---|---|---|---|
| **FP64 Compute** | **1335 GFLOPS** | 1430 GFLOPS | **93%** ✅ |
| FP32 Compute | 12364 GFLOPS | 22940 GFLOPS | 54% |
| **VRAM Bandwidth (sustained)** | **530 GB/s** | 608 GB/s | **87%** ✅ |
| PCIe H2D Transfer | 15.5 GB/s | ~50 GB/s | 31% (SYCL driver limit) |
| GPU Utilization (CFD) | **99%** | — | Full boost at 2800 MHz |
| Power (CFD load) | 181 W | 275 W max | 66% TDP |

**The B70 Pro delivers excellent raw performance for compute workloads.**
FP64 and VRAM bandwidth — the two metrics that matter most for CFD —
are both above 87% efficiency. This is better than many data center GPUs.

---

## Why GPU Acceleration Doesn't Win Yet — Software, Not Hardware

| Root Cause | Impact |
|---|---|
| **Level Zero kernel launch latency: ~100 µs** | 20× worse than CUDA (~5 µs) |
| **No GPU-aware MPI** | Host↔Device copy on every AllToAll iteration |
| **Ginkgo 1.10 SYCL: BJ only viable preconditioner** | Too weak for 34M-cell CFD pressure system |
| **Ginkgo 1.10 SYCL: IC/ICT/Multigrid crash or OOM** | No strong alternative |

The GPU computes correctly and fast — but the overhead *around* each
GPU operation dominates the total runtime. With 200 CG iterations per
pressure solve, ~100 µs per kernel launch adds up to seconds of pure
synchronization overhead per timestep.

> "It's not the GPU that's slow. It's the calls to the GPU."

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

| Improvement | Expected Effect | Timeline |
|---|---|---|
| Ginkgo 2.0 SYCL migration (OGL) | BJ maxBlockSize > 1, ParIC/ParICT available | When OGL migrates |
| Level Zero latency reduction | 5–10× less kernel overhead | 1–2 years |
| SYCL Graph in Ginkgo/OGL | Batch kernel launches → single overhead | 1–2 years |
| GPU-aware MPI for Xe | Eliminate host↔device copies | 2–3 years |

With **ParIC or Multigrid** as preconditioner (converging in 20–50 iter
instead of never), and **lower kernel latency**, GPU would likely win
by 2–5× over CPU for this mesh size.

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
| OGL (OpenFOAM-Ginkgo Layer) | dev branch |
| Ginkgo | 1.10 (OGL-internal) |
| Intel oneAPI | 2026.0.0 |
| Intel Compute Runtime | 26.05.37020 |

---

## Bugs Found & Documented (All New)

| # | Bug | Impact |
|---|---|---|
| [01](findings/01_pcie_reporting_bug.md) | lspci reports PCIe 1.0×1 (xe SR-IOV PF bug) | Diagnostic confusion |
| [02](findings/02_bj_maxblocksize_oom.md) | Ginkgo 1.10 SYCL: BJ maxBlockSize>1 → OOM | No strong BJ available |
| [03](findings/03_preconditioner_subdict_syntax.md) | OGL preconditioner sub-dict syntax undocumented | Options silently ignored |
| [04](findings/04_sycl_device_selector.md) | ONEAPI_DEVICE_SELECTOR: level_zero:0 not level_zero:gpu:0 | Crash on startup |
| [05](findings/05_sycl_preconditioner_status.md) | IC/ICT NotImplemented/DEVICE_LOST on SYCL | No ILU family available |
| [08](findings/08_multigrid_device_lost.md) | Multigrid OOM in PGM coarsening + divergence | No GPU multigrid |
| [09](findings/09_pcie_nvtop_confirmation.md) | nvtop confirms real PCIe GEN 5@16x (lspci wrong) | lspci bug confirmed |
| [10](findings/10_ginkgo2_api_breaks.md) | Ginkgo 2.0 API breaks OGL (3 locations) | Cannot upgrade |

---

## Repository Structure

```
├── README.md              — This file
├── hardware.md            — Full hardware specs and measured performance
├── setup/
│   ├── install_stack.md   — OGL/Ginkgo/oneAPI installation + required patches
│   └── bios_settings.md   — BIOS optimization for compute workloads
├── benchmarks/
│   ├── results.md         — All CFD benchmark results
│   └── hardware_results.json — Machine-readable hardware metrics
├── findings/              — Documented bugs (01–10)
└── configs/               — Working fvSolution configurations
```

---

## Status: May 2026

**Current recommendation:** Use CPU GAMG for production CFD on this hardware.
Reserve GPU for LLM inference (32 GB VRAM is excellent for this).

**Watch for:** OGL Ginkgo 2.0 migration, SYCL Graph support, Level Zero
latency improvements. Re-evaluate in 12–18 months.

*Co-documented with Claude (Anthropic) over an extended debugging session.*
*Full reproducibility intended for the next Battlemage CFD pioneer.*
