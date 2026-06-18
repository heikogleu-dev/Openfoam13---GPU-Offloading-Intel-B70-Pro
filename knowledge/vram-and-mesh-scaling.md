# VRAM & mesh scaling on the 32 GB B70

## Per-preconditioner memory cost (Ginkgo 2.0 standalone, `findings/26`)

| Preconditioner | bytes / row | fits at 34M? | fits at 7.1M? |
|---|---|---|---|
| BJ(1) | ~48 | yes (but too weak) | yes |
| BJ(8) | ~132 | (blocked by find_blocks anyway) | (blocked) |
| ILU | ~268 | **no** (OOM) | yes (~10.7 GB) |
| Multigrid (PGM) | ~1027 | no (~35 GB > 32) | *test pending* |

Standalone bytes/row understates the real OGL distributed footprint
(matrix + vectors + Schwarz + the factorization's temporaries).

## Measured peak VRAM (OGL distributed, fdinfo sum over ranks)

| Case | preconditioner | np | peak VRAM | outcome |
|---|---|---|---|---|
| 34M | ILU | 8 | **31.5 GB** | DEVICE_LOST in `Csr::convert_to(Coo)` (ParIlu generate) |
| 30.5M (Oem30) | ILU | 8 | **31.5 GB** | 1 pressure solve completed (181 iter), then DEVICE_LOST |
| 7.1M (half-res) | ILU | 8 | **10.7 GB** | runs to completion ✓ |

Key mechanism: the **`Csr::convert_to(Coo)` spike inside `ParIlu`
generate is largely mesh-size-insensitive** — 34M and 30.5M both peak at
~31.5 GB. So shaving ~11% off the cell count does **not** buy enough
headroom; you need a substantially smaller mesh.

## Practical ceilings (32 GB hardware, ~31.5 GB usable)

- **ILU:** needs roughly **< 25M cells** to run sustained at 32 GB on this
  stack (7.1M is comfortable at 10.7 GB; 30.5M is over the edge).
- **Multigrid:** ~1027 bytes/row → ~35 GB at 34M (won't fit); at 7.1M it
  *might* — that is the open test.
- **Desktop on the B70 eats ~1.15 GB** (gnome-shell + Xwayland), which
  directly reduces the headroom — relevant after the iGPU-PRIME removal
  (see hardware-system-grub.md).

## Multigrid VRAM ceiling on the 32 GB B70

Measured (7.1M, np=8, fdinfo incl. ~1.15 GB desktop): MG with a Jacobi
smoother ≈ **10.5–11.5 GB** → **~1.5 GB / million cells** (total),
~1.46 GB/M for the solver alone. SSOR smoother is the outlier at **26.5 GB**
(avoid). Usable ≈ 32 − ~1.15 (desktop) − ~0.5 (reserve) ≈ **30.3 GB**.

→ **Ceiling ≈ 20–21M cells** for the good MG config (V-cycle / Jacobi /
CG-coarse), double precision, np=8. Caveats:
- Could be a bit **higher** if per-cell overhead amortizes at scale (ILU
  dropped 1.55→1.03 GB/M from 7.1M→30.5M); but MG's Galerkin coarse
  operators add memory that grows too. Only a run at ~15M confirms it.
- **np-dependent:** fewer ranks = less Schwarz overlap = less VRAM (ILU went
  9.6→11.9 GB for np 2→12). At np=2 the ceiling is higher (~24M) but
  wall-clock worse; at np=12 lower (~18M).

## Levers to make VRAM leaner (ranked by impact)

1. **Mixed precision DP-SP (biggest — but needs an OGL patch).** FP64 finest
   vectors + FP32 coarse levels ~halves the coarse-hierarchy + matrix memory
   → ceiling could rise to ~30M+. Ginkgo supports it; **OGL only exposes
   `precision` for BJ (Preconditioner.hpp:172), not for Multigrid** — so it
   requires wiring mixed precision into OGL's Multigrid path. Research note:
   keep vectors higher precision than the matrix; **skip FP16** on Intel for
   the short-row pressure Laplacian (subgroup {16,32} → FP16 SpMV can be
   slower). See [gpu-amg-reference-configs.md](gpu-amg-reference-configs.md).
2. **Fewer MPI ranks (np 8→4).** Less Schwarz halo duplication → ~1–2 GB
   saved; costs some wall-clock (np=4 was ~26 s vs np=8 ~23 s for ILU).
   Available now, pure config.
3. **Leaner MG config.** `deep-coarse` (10.8 GB) slightly under `CG-coarse`
   (11.5 GB); **never SSOR** (26.5 GB, 2.5×). Already the chosen default.
4. **Desktop off the B70** (monitor → iGPU outputs): frees ~1.15 GB (~0.75M
   cells). Not available remotely (no monitor re-plug).
5. **Matrix format** is already CSR (the compact choice); ELL would pad, COO
   uses 3 arrays — no gain.

## Projection: can 34M fit with mixed precision, and would it win? (EXTRAPOLATION — confirm at 18M)

Extrapolated from the single measured point (MG CG-coarse double np=8 = 1.46
GB/M solver). **These are estimates with wide error bars; the 18M run will
calibrate them.**

| Mesh | double | DP-SP (~0.78×) | FP32 (~0.6–0.7×) |
|---|---|---|---|
| 7.1M | 10.3 GB ✓ | 8.1 GB ✓ | 6–7 GB ✓ |
| 18M | 26 GB ✓ | 20.5 GB ✓ | 16–18 GB ✓ |
| **34M** | **50 GB ✗** | **39 GB ✗** | **30–35 GB ⚠ marginal** |

- **34M VRAM verdict:** does **not** fit in double or DP-SP. **FP32-throughout
  is the only path, and it's marginal** — ~30–35 GB at np=8 (over the ~31.5 GB
  ceiling once the ~1.15 GB desktop is counted). It *might* fit at **np≤4** (less
  Schwarz overlap, ~−10–15%) and/or with the desktop off the B70. Realistic FP32
  ceiling is ~**25–28M** on a single 32 GB B70; 34M is at the absolute edge.
- **Performance IF it fit (34M):** 34M is well past the ~10M-cells/GPU breakeven,
  so the GPU is properly fed (compute util should rise well above the 46% seen at
  7.1M). CPU GAMG baseline at 34M ≈ **35.7 s/step** (measured, np=16). Projected
  GPU MG: double ≈ 25–40 s/step (util-improved); **FP32 ≈ 15–25 s/step** (FP32
  gives up to ~2.3× on the bandwidth-bound SpMV, ~7 nnz/row, Loe model
  5w/(2w+1)). → **plausibly a ~1.4–2× win over CPU GAMG at 34M, *if* VRAM allows.**
- **Practical target:** ~**20–25M with FP32** is the sweet spot — comfortably
  fits, well-fed GPU, likely the first clear GPU win. 34M is a stretch goal that
  needs FP32 + low rank count (+ desktop off B70), or a 2nd GPU / 48–64 GB card.

## How VRAM was measured (no debugfs / no sudo needed)

debugfs `vram_mm` needs a root `chmod` after each boot (can't do remotely).
Instead sum per-process drm memory from `/proc/<pid>/fdinfo/*`
(`drm-total-vram0` / `drm-resident-vram0`) over the `foamRun` ranks — used
in `gpu-diag/run-ilu-monitored.sh` and `precond-vram-sweep.sh`.

## Mixed precision PROTOTYPED (standalone Ginkgo, 2026-06-18)

`gpu-diag/diag-mixedmg.cpp` — Poisson 2.25M, MG-CG, measured device free-memory:
| mode | iters | VRAM | saving |
|---|---|---|---|
| double | 89 | 2107 MiB | — |
| DP-SP (double finest, float coarse) | 89 | 1800 | −16% |
| all-float preconditioner | 87 | 1583 | −26% |
| **fullfloat (matrix+vectors+MG all float)** | 68* | **1466** | **−30%** |

Mixed precision **compiles, runs, converges with no accuracy/iteration penalty**
in Ginkgo SYCL on the B70. Key: DP-SP / all-float-precond keep the *finest*
SpMV in double (the bandwidth bottleneck + the dominant matrix), so they give
little bandwidth benefit and don't fit 34M (~37 GB). **Only the full-float solve
(~1.0 GB/M in OGL terms) delivers both the bandwidth win and brings 34M into
range (~30–35 GB, fits at np≤4).** In OGL the precond change is contained
(Preconditioner.hpp); the full-float solve is a multi-file change (lduLduBase +
CG + MatrixWrapper, all `scalar`-templated).
