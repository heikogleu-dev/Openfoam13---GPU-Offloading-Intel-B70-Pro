# Hardware Details

## GPU: Intel Arc Pro B70 Pro

| Spec | Value |
|---|---|
| Architecture | Battlemage (Xe2-HPG) |
| Die | BMG-G31 (full Big Battlemage) |
| Xe Cores | 32 |
| XMX Engines | 256 |
| Boost Clock | 2800 MHz |
| Graphics Clock | 2280 MHz (typical) |
| VRAM | 32 GB GDDR6 ECC |
| Memory Bus | 256-bit |
| Memory Bandwidth | 608 GB/s (spec), 508 GB/s measured D2D |
| TDP | 230 W |
| PCIe | 5.0 x16 |
| FP64 | ~1.43 TFLOPS |
| FP32 | ~22.94 TFLOPS |
| Release | March 2026 |

## PCIe Topology (Important!)

```
CPU PEG 00:06.0          → PCIe 5.0 ×16 ✅ (correct)
└─ On-Card Switch 02:00.0 → PCIe 5.0 ×16 ✅ (correct)
   └─ Switch DS 03:01.0   → PCIe 1.0 ×1  ← REPORTING BUG (xe driver)
      └─ B70 GPU 04:00.0
```

**`lspci` reports PCIe 1.0 x1 — this is a known xe-driver bug in SR-IOV PF mode.**

Real bandwidth measured via SYCL memcpy: ~15 GB/s H2D (consistent with PCIe 4.0 x8
due to dual-tile architecture splitting the link internally; PCIe 1.0 x1 would
be 0.25 GB/s — 60× slower than measured).

See [findings/01_pcie_reporting_bug.md](findings/01_pcie_reporting_bug.md).

## Measured Hardware Performance

| Test | Result | Theory | Efficiency |
|---|---|---|---|
| **GPU FP64 FMA** | **1335 GFLOPS** | 1430 GFLOPS | **93%** |
| GPU FP32 FMA | 12364 GFLOPS | 22940 GFLOPS | 54% |
| **VRAM Triad sustained (2 GiB)** | **531 GB/s** | 608 GB/s | **87%** |
| VRAM D2D Triad (4 MB) | 508 GB/s | 608 GB/s | 83% |
| PCIe H2D (64MB) | 15.5 GB/s | ~50 GB/s | 31% |
| RAM Triad (8 P-Cores) | 77 GB/s | 109 GB/s | 71% |
| CPU DGEMM FP64 (8T MKL) | 592 GFLOPS | ~1500 GFLOPS | 39% |
| CPU DGEMM FP64 (24T MKL) | 1106 GFLOPS | ~1500 GFLOPS | 74% |

> **GPU FP64 hits 93% of theoretical peak** — the silicon is excellent for
> double-precision compute. The FP32 spec (22.94 TFLOPS) assumes XMX-style
> packed math; pure SIMD FMA gets 1× per ALU.
>
> **VRAM Triad ~530 GB/s is sustained** from 64 MiB up to 4 GiB per array
> (12 GiB total touched). The 16 MiB outlier (1672 GB/s) is an SLC cache hit.

## PCIe H2D Bandwidth vs Transfer Size

| Transfer Size | H2D GB/s | D2H GB/s | D2D GB/s | Latency µs |
|---|---|---|---|---|
| 1 MB | 10.4 | 14.4 | 55.1 | 101 |
| 4 MB | 12.3 | 12.6 | 508.0 | 341 |
| 16 MB | 14.1 | 15.9 | 273.5 | 1191 |
| 64 MB | 15.5 | 14.4 | 257.6 | 4338 |
| 256 MB | 14.8 | 13.7 | 259.8 | 18105 |
| 1024 MB | 9.9 | 6.1 | 259.0 | 108310 |

> **Key finding:** PCIe bandwidth is NOT the bottleneck for our CFD case.
> Level Zero kernel launch latency (~100 µs per operation) is.
