# B70 vs current AMD / NVIDIA / Intel GPUs (for bandwidth-bound CFD)

Researched 2026-06-18, cited. Prices are June-2026 street estimates (volatile —
GDDR shortage + AI demand). Standard (non-tensor) throughput unless noted.

| GPU | Mem BW (GB/s) | VRAM | FP64 std (TF) | FP32 (TF) | ~Price $ | GB/s per $ | VRAM-GB per $ |
|---|---|---|---|---|---|---|---|
| **Intel Arc Pro B70** | **608** (spec; 530 meas. sustained) | **32 GB** | ~1.3 (meas.) | 22.9 | **~950–1083** | **~0.59** | **~0.031** ★ |
| NVIDIA RTX 4090 | 1008 | 24 GB | 1.29 (1:64) | 82.6 | ~2400 | ~0.42 | ~0.010 |
| NVIDIA RTX 5090 | 1792 | 32 GB | 1.64 (1:64) | 104.8 | ~4000 | ~0.45 | ~0.008 |
| NVIDIA L40S | 864 | 48 GB | ~1.4 (1:64) | 91.6 | ~8000+ | ~0.10 | ~0.006 |
| NVIDIA A100 80GB | 1935–2039 | 80 GB | **9.7** | 19.5 | ~15k (used less) | ~0.12 | ~0.005 |
| NVIDIA H100 SXM | **3350** | 80 GB | **34** | 67 | ~30–40k | ~0.10 | ~0.002 |
| AMD RX 7900 XTX | 960 | 24 GB | 1.92 (1:32) | ~46 | ~800–1300 | ~0.85 | ~0.022 |
| AMD Radeon Pro W7900 | 864 | 48 GB | ~1.9 | ~46 | ~3400 | ~0.24 | ~0.013 |
| AMD Instinct MI210 | 1638 | 64 GB | **22.6** | 22.6 | ~6–10k | ~0.20 | ~0.008 |
| AMD Instinct MI300X | **5300** | **192 GB** | **81.7** | 163.4 | ~15–18k | ~0.32 | ~0.012 |
| Intel Max 1100 | 1229 | 48 GB | **22.2** | 21.8 | ~3–4k | ~0.35 | ~0.013 |
| Intel Max 1550 | **3277** | 128 GB | **52.4** | 51.4 | ~8–10k | ~0.39 | ~0.014 |

## Verdict for our use case (32 GB, bandwidth-bound, mixed precision)

- **B70 is the clear winner on VRAM-per-dollar among 32GB+ cards** (~2.5–4× better
  than RTX 5090 / W7900 / MI300X / Max 1550). The cheapest way to get 32 GB at
  decent bandwidth, with native oneAPI/SYCL fit. **Sweet spot for large-memory,
  bandwidth-efficient FP32/FP16 explicit CFD (LBM/FluidX3D) on a budget.**
- **Bandwidth** (608 GB/s spec) is mid-pack consumer-class (≈L40S, below 7900XTX/
  4090); 3.3× behind MI300X, 5.5× behind H100 SXM. GB/s-per-$ competitive (~0.59).
- **FP64** is weak (~1.3 TF, like all consumer cards) — strong FP64 is data-center
  only (MI300X 82, Max 1550 52, H100 34, A100 9.7). **But our diagnostics show FP64
  is NOT the bottleneck** (bandwidth + the SYCL preconditioner software are).
- **LBM efficiency: B70 hits 85% of its bandwidth** (FluidX3D) — ties RTX 4090,
  beats Intel's own Max 1100 (47%). For FP32/FP16 LBM the B70 is genuinely good.
- **No-compromise upgrade** = AMD **MI300X** (3.3× bandwidth, 192 GB, full FP64)
  at ~16–19× the price; **A100 80GB** = meaningful bandwidth+FP64 step-up at
  sometimes-comparable used price but 2.5× less VRAM/$ and no SYCL-native fit.

Sources: TechPowerUp, Intel ARK ([B70](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)),
FluidX3D README, Lehmann et al. PRE 2022 ([arXiv:2112.08926](https://arxiv.org/pdf/2112.08926)),
NVIDIA/AMD datasheets, [NVIDIA HPCG blog](https://developer.nvidia.com/blog/optimizing-high-performance-conjugate-gradient-benchmark-gpus/).

> Note: B70 bandwidth spec is **608 GB/s** (256-bit × 19 Gbps); our **530 GB/s**
> is measured *sustained* (87%). FP64 is vendor-unpublished; ~1.3 TF = our clpeak.
