# Confirmation: nvtop Shows Real PCIe GEN 5@16x

## nvtop Output (during CFD run)

```
Device 1 [Battlemage G31 (Intel Graphics)] PCIe GEN 5@16x
GPU 2800MHz  TEMP 64°C  FAN 1423RPM  POW 181W
GPU[99%]  MEM[ 8.541GiB/27.898GiB]
```

## What This Confirms

1. **PCIe 5.0 x16 is real** — nvtop reads from a different register than lspci
2. **lspci shows 1.0 x1** — confirmed xe driver SR-IOV PF mode reporting bug
3. **Usable VRAM: 27.9 GB** — not 32 GB due to ECC + SR-IOV PF overhead
4. **Display correctly on iGPU (Device 0)** — B70 Pro is pure compute

## GPU During CFD Run

| Metric | Value |
|---|---|
| PCIe | GEN 5@16x ✅ |
| GPU Utilization | 99% (100% during p-solve) |
| VRAM Used | 8.54 GiB / 27.9 GiB available |
| Temperature | 64°C |
| Power | 181 W / 275 W (66% TDP) |
| Boost Clock | 2800 MHz (full boost) |

## Idle After Multigrid Crash

```
Device 1 [Battlemage G31] PCIe GEN 5@16x
GPU 400MHz  TEMP 52°C  FAN 1186RPM  POW 55W
GPU[2%]  MEM[ 1.170GiB/27.898GiB]
```

GPU recovered automatically — no reboot required after DEVICE_LOST.
