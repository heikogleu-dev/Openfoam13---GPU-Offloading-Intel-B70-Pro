# CPU Tuning: No Performance Impact on Arrow Lake GAMG

## Tests

| Configuration | s/Step T=8-10 | vs Baseline |
|---|---|---|
| Baseline (madvise/swap60/powersave) | 35.9 | — |
| THP=always | 35.6 | −0.8% (noise) |
| THP=always + swap10 + performance governor | 36.2 | +0.8% (noise) |

## Verdict

All three runs statistically identical (±1.7% measurement noise).
Arrow Lake CPU GAMG performance is DDR5 bandwidth-bound at ~77 GB/s.
No software tuning can exceed this physical limit.

## Why Each Tuning Had No Effect

- **Governor performance vs powersave:** intel_pstate with
  `max_perf_pct=100` runs at full Turbo (4.7 GHz measured) regardless.
- **Swappiness=10:** 91 GB RAM, only ~7 GB used — swap never touched.
- **THP=always:** OpenFOAM allocates once at init, reuses memory.
  GAMG coarse levels fit in L2 cache — TLB pressure irrelevant.

## Implication

35.9 s/Step is the hard physical limit for CPU GAMG on this hardware.
GPU offloading must achieve < 35.9 s/Step with real convergence
to justify its complexity. This is the definitive CPU baseline.
