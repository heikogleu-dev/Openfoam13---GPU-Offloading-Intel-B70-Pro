# VRAM Analysis — Where Does Memory Actually Go?

## Methodology

VRAM sampled directly from xe driver debugfs at 5 Hz throughout the run.
This avoids the limitations of `intel_gpu_top` (i915-PMU only, not xe)
and `nvtop` (snapshot-only, no continuous trace).

```bash
# One-time setup (after each reboot — debugfs perms reset)
sudo chmod a+x /sys/kernel/debug /sys/kernel/debug/dri /sys/kernel/debug/dri/0 \
               /sys/kernel/debug/dri/0/tile0
sudo chmod a+r /sys/kernel/debug/dri/0/tile0/vram_mm

# Sampler loop
while true; do
    USAGE=$(grep "^  usage:" /sys/kernel/debug/dri/0/tile0/vram_mm | awk '{print $2}')
    echo "$(date +%s.%N),$USAGE"
    sleep 0.2
done
```

The full sampler script used in this work is preserved at
[`logs/vram-traces/`](../logs/vram-traces/) along with the resulting CSVs.

## Hardware

- Intel Arc Pro B70 Pro (BMG-G31, Battlemage Xe2)
- 32 GB GDDR6 raw
- 27.9 GB usable (after firmware reservation, per `vram_mm: size`)
- Idle baseline: ~1.2 GB (display, X server, ambient compositor)

## Measured BJ(1) Footprint — 34M cells, np=8

```
Trace duration:  220.8 s (1057 samples)
Baseline:        1.183 GB
Peak:            9.376 GB (at t = 78 s, during the first p-solve)
Δ over baseline: 8.192 GB
68% of run:      7-9 GB
```

## Comparison to KIT Estimate

The KIT/Ginkgo team (greole) estimated ~5.3 GB for the p-system matrix:

```
N_cells × NNZ_per_row × (T_scalar + 2×T_label + T_reorder)
  + N_cells × 2 × T_scalar
= 34M × 7 × 16 + 34M × 16
≈ 5.3 GB
```

Measured Δ is 8.2 GB — about 3 GB above estimate. Likely accounts for:

- p system matrix (CSR): ~5.3 GB ✅
- CG Krylov vectors (4-5 vectors × 8 bytes × 34M cells): ~1.4 GB
- Initial-guess + RHS scratch: ~0.3 GB
- BJ(1) preconditioner storage: ~0.3 GB
- Ginkgo workspace + alignment overhead: ~0.9 GB

**Important:** The measured 8.2 GB confirms greole's "tight coupling"
hypothesis is approximately correct — *only the p-equation* is on the
GPU, NOT all 5 fields (which would be 30+ GB and exceed VRAM).

## BJ(maxBlockSize=2) Crash Analysis

```
Trace duration:    74.8 s (359 samples)
Peak before crash: 8.41 GB (LESS than BJ(1)!)
Available:         27.9 GB
Crash type:        SIGABRT in preconditioner generate
```

The crash happens BEFORE the BJ(2) workspace is allocated — so the peak
only reflects matrix + setup, not the bug. See [`findings/02`](../findings/02_bj_blocksize_int_underflow.md)
for the actual root cause (`size_t` underflow, not memory pressure).

## Headroom Available for Stronger Preconditioners

With only 8 GB of 27.9 GB VRAM used by BJ(1), there is **19.9 GB free**
for stronger preconditioners. This means:

| Preconditioner | Estimated additional footprint | Fits in 19.9 GB free? |
|---|---|---|
| BJ maxBlockSize=4 | +0.5 GB workspace | ✅ Easily |
| BJ maxBlockSize=8 | +1.1 GB workspace | ✅ Easily |
| BJ maxBlockSize=16 | +2.2 GB workspace | ✅ Easily |
| ParIc factorization (L matrix) | +5-8 GB | ✅ Comfortably |
| Multigrid PGM (full hierarchy) | +10-15 GB | ✅ Probably fits |

**Conclusion:** Hardware is NOT the constraint. All currently failing
preconditioners would fit in available VRAM if the SYCL bugs were fixed.

## Files in `logs/vram-traces/`

- `vram-bj1.csv` — raw VRAM trace, 220 s, 1057 samples (BJ(1) baseline)
- `vram-bj2.csv` — raw VRAM trace, 75 s, 359 samples (BJ(2) crash)
- `log.vram-bj1` — full mpirun output for BJ(1) baseline
- `log.vram-bj2` — full mpirun output including SIGABRT stack trace
