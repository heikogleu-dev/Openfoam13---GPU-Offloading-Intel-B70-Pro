# Bug: lspci Reports PCIe 1.0 x1 (False)

## Symptom

```
$ lspci -vv -s 04:00.0 | grep LnkSta
LnkSta:  Speed 2.5GT/s, Width x1
```

Also `/sys/bus/pci/devices/0000:04:00.0/current_link_speed = 2.5 GT/s PCIe`
and `max_link_speed = 2.5 GT/s PCIe`.

## Root Cause

Intel Arc cards (Alchemist AND Battlemage) have an on-card PCIe switch:

```
00:06.0 (CPU PEG)               LnkCap: 32GT/s × 16  ✅
└─ 02:00.0 (Switch upstream)    LnkCap: 32GT/s × 16  ✅
   ├─ 03:01.0 (DS to GPU)       LnkCap:  2.5GT/s × 1 ❌  ← reporting bug
   │  └─ 04:00.0 (B70 Pro GPU)  LnkSta:  2.5GT/s × 1 ❌
   └─ 03:02.0 (DS to HDMI Audio)
      └─ 05:00.0 (Audio)
```

The `xe` driver in SR-IOV PF mode does not update the downstream LnkSta caps
correctly — they remain at the conservative power-on default.

## Proof It's a Reporting Bug, Not Real

Real bandwidth measured via SYCL malloc_device + memcpy benchmark:

| Transfer | Measured | PCIe 1.0×1 max | PCIe 4.0×8 max | Verdict |
|---|---|---|---|---|
| H2D 64 MB | 15.5 GB/s | 0.25 GB/s | ~16 GB/s | matches PCIe 4.0×8 |
| D2H 64 MB | 14.4 GB/s | 0.25 GB/s | ~16 GB/s | matches PCIe 4.0×8 |

15.5 GB/s is **60× faster than PCIe 1.0×1 allows** — the reported caps are
clearly wrong.

Windows Intel Graphics Control Panel correctly reports "PCIe 5.0 x16" for
the same hardware.

## Affected Systems

- Intel Arc Alchemist (A770, A750, A580, etc.)
- Intel Arc Battlemage (B70 Pro, B580, B570, etc.)
- All cards with the on-card PCIe switch (= all Arc dGPUs to date)

## Workaround

None for the lspci/sysfs reporting itself. **Don't trust `lspci LnkSta`** —
measure real bandwidth via SYCL/CUDA bench instead.

For quick verification:
```cpp
#include <sycl/sycl.hpp>
sycl::queue q{sycl::gpu_selector_v};
size_t N = 64*1024*1024 / 8;
double *d = sycl::malloc_device<double>(N, q);
double *h = new double[N];
auto t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 20; i++) q.memcpy(d, h, 64*1024*1024).wait();
auto t2 = std::chrono::high_resolution_clock::now();
// ... real bandwidth = (20 * 64 MB) / elapsed
```

## Reference

- nvtop GitHub PR #344 (Steve-Tech) documents same issue for Alchemist
- Confirmed still present in Battlemage on Linux 7.0 / xe 1.1.0 (May 2026)
- Bug-Report at `xe`-driver upstream might fix this — relevant only for
  monitoring tools, real performance is unaffected
