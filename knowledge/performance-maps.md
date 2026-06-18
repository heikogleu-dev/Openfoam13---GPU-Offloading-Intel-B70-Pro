# Performance maps: (preconditioner × ranks) → util / VRAM / iters / wall-clock

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
