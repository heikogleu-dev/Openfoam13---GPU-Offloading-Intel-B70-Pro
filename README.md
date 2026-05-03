# Intel Arc Pro B70 Pro + OpenFOAM CFD — Pioneer Documentation

> **TL;DR:** First documented attempt to run OpenFOAM GPU acceleration
> (OGL/Ginkgo/SYCL) on Intel Arc Pro B70 Pro (Battlemage) for automotive
> CFD. Honest results, real bugs found, practical fixes documented.

## Hardware

| Component | Spec |
|---|---|
| GPU | Intel Arc Pro B70 Pro (Battlemage G31, 32 Xe-Cores) |
| VRAM | 32 GB GDDR6 ECC, 608 GB/s |
| CPU | Intel Core Ultra 9 285K (8P+16E Cores) |
| RAM | 96 GB DDR5-6800 |
| Mainboard | ASRock Z890I Nova WiFi |
| PCIe | 5.0 x16 (physical) |
| OS | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-15-generic |

## Software Stack

| Component | Version |
|---|---|
| OpenFOAM | Foundation 13 (build 13-441953dfbb42) |
| OGL (OpenFOAM-Ginkgo Layer) | dev branch (internal Ginkgo 1.10) |
| Ginkgo | 1.10 (OGL-internal) + 2.0 develop (/opt/ginkgo) |
| Intel oneAPI | 2026.0.0 |
| Intel Compute Runtime | 26.05.37020 |
| Level Zero | 1.28.2 |
| xe Driver | 1.1.0 (kernel 7.0) |
| SYCL | icpx 2026.0.0 |

## Case: MR2 SW20 Vehicle Aerodynamics

- **Mesh:** 34,088,573 cells (31M hex + 2.4M polyhedra)
- **Solver:** simpleFoam, k-omega SST RANS
- **Reference area:** 1.857 m²
- **Vehicle:** Toyota MR2 SW20, 1050 kg, 850 WHP

## Key Results

| Configuration | s/Step (Steady-State T=8-10) |
|---|---|
| GPU np=2, BJ default | 96.0 |
| GPU np=8, BJ default | 52.0 |
| **CPU GAMG np=16 (P+E Cores)** | **35.7** ← Winner |
| GPU np=16, BJ optimized (nNonOrth=1, maxIter=80) | 30.6 |
| GPU np=16, BJ default, nNonOrth=2 | ~50.0 |

> **Verdict:** CPU GAMG with np=16 wins at equal solver settings.
> GPU only beats CPU with aggressively reduced quality settings.
> Root cause: Level Zero kernel launch latency (~100 µs) + no GPU-aware MPI.

## Quick Navigation

- [Hardware Details](hardware.md)
- [Installation Guide](setup/install_stack.md)
- [BIOS Settings](setup/bios_settings.md)
- [Benchmark Results](benchmarks/results.md)
- [Bugs & Findings](findings/)
- [fvSolution Configs](configs/)
- [Conclusions](conclusions.md)
