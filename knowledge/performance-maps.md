# Performance maps: (preconditioner × ranks) → util / VRAM / iters / wall-clock

## ★ PLUGIN CEILING VERDICT (2026-06-26) — 1.18× is the practical max, cheap levers give ~0
Measured the cheap tuning levers at 17.2M mixed (baseline 18.73 s/step = 1.18× vs CPU-GAMG):
**D2H copy-offload-off = −4.5% (worse); renumberMesh RCM = 0 (band −22.5× but CPU not
cache-bound); decomp tuning = no headroom (scotch already 0.77% inter-rank faces); AMG-reuse
already in baseline (~1.12×).** → The plan's 1.45-1.5× is NOT reachable via tuning. Bottleneck
= ~64% CPU-rest (assembly/U-k-omega/MPI, neither cache- nor halo-bound) + algorithmic GPU-AMG
(bandwidth-bound ≈ roofline). Past 1.18× needs architecture: GPU-resident assembly (NeoN),
GPU-aware MPI (#922, now CR-26.22-unblocked), RS-AMG/Chebyshev (Ginkgo-side). Full analysis:
`findings/plugin-max-session/RESULTS.md`. renumberMesh adopted as standard prep anyway (hygiene).



Measured with `gpu-diag/compare-mesh.sh <case>` (np=8 = ranksPerGPU 8;
CR 26.05; from uniform init; tolerance 1e-6, relTol 0.01; steady = step2→3
ExecutionTime delta). GPU util = compute-engine (drm-cycles-ccs) busy %.

## 7.1M mesh (Testcase-half)

MG V+CG-coarse rank curve:
| np | VRAM | util | iters | s/step |
|---|---|---|---|---|
| 2 | 9.4 GB | 21% | ~12 | 18.0 |
| 4 | 10.0 GB | 34% | ~12 | 12.2 |
| 8 | 11.4 GB | 44% | ~12 | 9.0 |
| 12 | 11.0 GB | 38% | ~12 | 8.9 |
| 16 | 13.3 GB | 43% | ~12 | 8.5 |

Preconditioners @np=8: MG-VcgC 9.0s (44%, ~12 it) · MG-WcgC 12.7s (~8 it) ·
MG-default 14.0s (~64 it) · ILU 24.0s (~201 it) · **GAMG-CPU 9.3s (np=8), 8.5s (np=16)**.

## 17.2M mesh (Testcase-mid)

MG V+CG-coarse rank curve:
| np | VRAM | util | iters | s/step |
|---|---|---|---|---|
| 2 | 19.6 GB | 23% | ~13 | 47.7 |
| 4 | 22.9 GB | 32% | ~13 | 33.1 |
| 8 | 21.0 GB | 40% | ~13 | 24.2 |
| 12 | 22.9 GB | 45% | ~13 | 22.8 |
| 16 | 22.6 GB | 44% | ~13 | 22.7 |

Preconditioners @np=8: MG-VcgC 24.2s (40%, ~13 it) · MG-WcgC 34s (~8 it) ·
MG-default 38.5s (~64 it) · ILU 60s (~201 it, capped) · **GAMG-CPU 25.9s (np=8), 22.1s (np=16)**.

## Single-precision MG (our OGL mixed-precision patch, `precision single`)

17.2M MG V+CG-coarse, `precision single` (all-float preconditioner) rank curve:
| np | VRAM | util | s/step | vs double | vs GAMG |
|---|---|---|---|---|---|
| 2 | 14.4 GB | 20% | 45.9 | −4% | — |
| 4 | 14.8 GB | 28% | 31.4 | −5% | — |
| 8 | 17.5 GB | 36% | 22.4 | −7% | **GAMG 25.9 → −14%** |
| 12 | 17.5 GB | 38% | 21.0 | −8% | — |
| 16 | 17.5 GB | 38% | **20.9** | −8% | **GAMG 22.1 → −6%** |

→ **🏆 GPU single-precision MG BEATS CPU GAMG at 17.2M, at every rank count**
(np=16: 20.9 vs 22.1; np=8: 22.4 vs 25.9) — and via just the *preconditioner*
patch, no full-float solve yet. Iters identical to double (~13, no accuracy
penalty). VRAM −17% (np=8, 21→17.5 GB) to −27% (np=2). 7.1M: single 7.95 s/step
also beats GAMG (8.5–9.3).

**Net with mixed precision: the GPU clearly wins at ≥17M.** The double-precision
near-tie is broken by `precision single`. VRAM ceiling lifts ~20M→~28-30M; 34M
still needs the full-float *solve* (CG matrix stays double).

## Verdict (double precision)

- **GPU-MG ≈ CPU-GAMG, a near-tie across 7–17M.** GPU wins at np=8 (+3–7%),
  CPU wins at np=16 (+3%) because the 24-thread i9-285K scales GAMG well while
  the single GPU saturates at ~np=12. No clear GPU win in double precision.
- **Still CPU/transfer-bound even at 17.2M:** util only ~40–45% (rises weakly
  with mesh size). The "feed the GPU" effect is real but modest in double — the
  GPU is not saturated; the limiter is host assembly + the ~30% copy engine.
- **Iteration count is mesh- and rank-independent** for MG-VcgC (~12–13) — good
  multigrid behaviour. ILU caps at 201 (too weak); MG-default ~64; W-cycle ~8
  (fewest, but W's per-cycle cost makes wall-clock worse than V+CG-coarse).
- **VRAM scales ~linearly** (~1.2 GB/M at np=8); 17.2M ≈ 21 GB (fits). VRAM
  rises with np (Schwarz overlap): 7.1M 9.4→13.3 GB for np 2→16.

**Implication:** in double precision the B70 only *matches* CPU GAMG. The clear
win must come from **FP32** (halves the bandwidth-bound SpMV that limits us) —
which also unlocks the 34M mesh (VRAM). That is the next step. See
[vram-and-mesh-scaling.md](vram-and-mesh-scaling.md),
[preconditioners-and-gpu-cfd.md](preconditioners-and-gpu-cfd.md).

## Performance projection (2026-06-18) — PROJECTED, not measured

Built on the measured baseline + audit-corrected lever sizes. **Baseline = measured;
everything below = projected (uncertainty flagged).** 17.2M, np16, s/step:

| Stage | s/step | vs CPU GAMG (22.1) | basis |
|---|---|---|---|
| **Baseline today** (single-MG) | **20.9** | 1.06× | ✅ measured |
| + C (AMG values-only reuse) | ~16–17 | ~1.3–1.4× | projected |
| + C + D + tuning (full-float, copy-engine/batching) | ~14–15 | ~1.4–1.6× | projected |
| **Hard ceiling** (GPU p-solve → 0) | **~11.5** | ~1.9× | Amdahl |

**Logic:** the GPU does only the pressure-solve (~40–48% of wall-clock); the other
~52–60% is CPU (U/k/omega DILU solves + p-assembly) — GAMG runs that same CPU work.
- **C** ~2× on the GPU p-solve → ~15–20% wall-clock → ~16–17 s.
- **D (full-float)** is mainly a **VRAM** lever (enables 34M); ~5–10% extra bandwidth.
- **Tuning A/Bs** (V1-batching, copy-engine, clock-pin): ~5% combined, uncertain.

**The real ceiling:** even with GPU p-solve → 0, the step floors at ~11.5 s (the CPU
half) = ~1.9× vs GAMG. Beating that needs offloading U/k/omega + assembly too (a
separate, large project) — the true architectural lever.

**Two nuances:** (1) the advantage **grows with mesh size** (B70 under-fed at 7.1M,
~46% busy; better-fed at 17–34M → bigger per-cell margin over GAMG; at 34M+full-float
the margin likely widens). (2) This is **not** FluidX3D/LBM territory (~6750 MLUPS,
GPU-native) — implicit FVM pressure-solve is a different method; the goal is beating
CPU GAMG + freeing CPU cores.

**Uncertainties:** C's payoff depends on the value-refill cost + iteration-creep
(untested); D's bandwidth gain is estimated; the CPU floor is firm.

## ★ AMG-reuse (Plan C) — MEASURED + WORKING (2026-06-19)

The AMG values-only reuse port works. Caching A/B, 7.1M single np=8 (same session):
| caching | init_precond | s/step | convergence |
|---|---|---|---|
| 0 (full rebuild) | 486 ms | 8.00 | 12–15 iters |
| 1 | 356 ms (−27%) | 7.67 (−4%) | healthy |
| **2 (sweet spot)** | **309 ms (−36%)** | **7.33 (−8.4%)** | healthy |
| 3 | 295 ms (−39%) | 7.33 (−8.4%) | healthy |

**No convergence penalty** (iters identical to full rebuild). Reuse-solve init_precond
≈ 363 ms vs build ≈ 670 ms (−46% per cache-hit). Plateau at caching=2–3.
**GPU single-MG + C at 7.1M = 7.33 s/step vs CPU GAMG 8.5–9.3 → ~1.2–1.3× faster**
(was ~1.1× without caching). At 17.2M the wall-clock gain should be larger (GPU
p-solve is a bigger fraction there) — measurement pending. This replaces the earlier
projection for C with measured data; the ~15–20% projection was optimistic at 7.1M
(~8% measured) but may hold at 17.2M.

### C at 17.2M (np16 single) — bigger gain than at 7.1M (2026-06-19)
| caching | init_precond | s/step | vs CPU GAMG (22.1) |
|---|---|---|---|
| 0 (full rebuild) | 1646 ms | 21.00 | 1.05× |
| **2** | **924 ms (−44%)** | **18.67 (−11.1%)** | **~1.18×** |
Iters identical/healthy. As predicted, the wall-clock gain is larger at 17.2M
(−11%) than 7.1M (−8%) because the GPU pressure-solve is a bigger fraction of the
step. **New GPU best at 17.2M = 18.67 s/step (single-MG + AMG-reuse) → ~1.18× faster
than CPU GAMG** (was 20.9 = 1.06× without caching). The audit projection (~16–17 /
~1.3×) was a touch optimistic; measured ~1.18×.

## ★ Kartierung 17.2M MIT caching + the CPU-bound finding (2026-06-19)

Rank map, GPU single-MG + caching=2 vs CPU GAMG:
| ranks | GPU single-MG+C | CPU GAMG | speedup |
|---|---|---|---|
| np8  | 20.33 | 25.00 | **1.23×** |
| np16 | 18.67 | 22.10 | **1.18×** |

Higher caching (np16): caching=2 → 19.0/init_precond 923ms; caching=10 → 18.3/705ms
(marginal extra, no iter-creep). C plateaus ≈ 18.0 s/step (~1.23× GAMG).

**★★ THE key finding — GPU is only ~30% utilized (compute-util ccs=31% @caching=2,
29% @caching=10; copy-util bcs ~9-10%).** The workload is **CPU-BOUND**: the GPU sits
idle ~70% of the time waiting on the CPU half (U/k/omega DILU solves + matrix assembly
≈ 49% of the step). This is **why C is capped at ~11-13%** — optimizing a GPU phase
(AMG setup) frees GPU time that was already idle. The big lever is feeding the GPU
(offloading the CPU half), gated by `call_init` (matrix construction) — see
per-iteration-diagnostics.md.
