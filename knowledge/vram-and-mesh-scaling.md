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

## How VRAM was measured (no debugfs / no sudo needed)

debugfs `vram_mm` needs a root `chmod` after each boot (can't do remotely).
Instead sum per-process drm memory from `/proc/<pid>/fdinfo/*`
(`drm-total-vram0` / `drm-resident-vram0`) over the `foamRun` ranks — used
in `gpu-diag/run-ilu-monitored.sh` and `precond-vram-sweep.sh`.
