# Finding 17: Hybrid CPU/GPU Solver Strategy

## Hypothesis

With the pressure equation (elliptic) requiring Multigrid for fast
convergence, keep `p` on CPU GAMG. Only offload `U`/`k`/`ω` (parabolic,
weak preconditioner sufficient) to GPU. This avoids the Ginkgo SYCL
preconditioner gap entirely — best of both worlds.

## Configuration

```
p:       GAMG / CPU         (5–10 iter typical)
U/k/ω:   GKOBiCGStab / GPU + BJ(maxBlockSize=1)
```

## Result

All fields converge cleanly:

| Field | Solver | Typical iters | Final residual |
|---|---|---|---|
| p (CPU) | GAMG + DICGS | 3–5 | < 1e-5 |
| U_{x,y,z} (GPU) | GKOBiCGStab + BJ(1) | 2 | < 1e-3 |
| k (GPU) | GKOBiCGStab + BJ(1) | 3–4 | < 1e-3 |
| ω (GPU) | GKOBiCGStab + BJ(1) | 1–2 | < 1e-3 |

**But the per-step time is worse than CPU-only:**

| Configuration | s/step (T=4,5 mean) | vs CPU GAMG np=16 baseline | vs GPU BJ(1) only |
|---|---|---|---|
| CPU only (GAMG, np=16) | **35.9** | — | — |
| GPU only (BJ p, np=8) | 53.3 | +49 % slower | — |
| **Hybrid (CPU GAMG p + GPU U/k/ω, np=8)** | **48.0** | **+34 % slower** | **−10 % faster** |

Init phase (T=1): **176 s** — nearly 2× the GPU-only init cost (96 s).
Five additional SYCL executor initialisations (U_x, U_y, U_z, k, ω) add
significant startup overhead.

## Interpretation

Hybrid is **slightly faster than full GPU**, but **substantially slower
than full CPU**. Three reasons:

1. **U/k/ω were not the bottleneck** — they already converge in 1–4 iter
   on CPU PBiCGStab+DILU. Moving them to GPU adds host-buffer-copy and
   executor overhead without removing meaningful work.
2. **Doubled init cost** for 4 GPU-side executors (U_x, U_y, U_z, k, ω).
3. **Same MPI / PCIe bottleneck** as documented in
   [profiling/bottleneck_analysis.md](../profiling/bottleneck_analysis.md):
   `forceHostBuffer=true` is still needed for the U/k/ω solves, so every
   halo exchange still round-trips through host memory.

## Verdict

**No net gain.** Hybrid does not enable GPU acceleration for the *hard*
problem (pressure). It adds GPU overhead for the *easy* problems
(momentum / turbulence) which were never CPU-bound.

The 34M-cell mesh remains GAMG-bound on CPU. GPU U/k/ω offload is a
distraction — the question is whether GPU can do `p` faster, and the
answer remains "not with Ginkgo 1.10 SYCL."

## Status

This concludes the parameter-tuning exhaustion phase. **All plausible
fvSolution configurations have been tested:**

- BJ maxBlockSize 1 / 2 / 4 / 8 / 16 (only BS=1 stable)
- ISAI sparsityPower 1 / 3 (with and without `scaling=-1.0`)
- IC, ICT (NotImplemented / DEVICE_LOST)
- Multigrid (OOM + diverge)
- Hybrid (CPU p + GPU U/k/ω) — this finding
- BiCGStab vs CG, evalFrequency, splitComm
- np 4 / 8 / 16 with matching ranksPerGPU

Further improvement requires upstream OGL/Ginkgo work — see
[README.md](../README.md) "When to Re-evaluate GPU Offloading".
