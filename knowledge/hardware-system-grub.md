# Hardware, system & GRUB

## GPU / host

- **Intel Arc Pro B70** — BMG-G31 (Battlemage Xe2), 32 GB GDDR6,
  PCI `04:00.0`, `renderD129`, `level_zero:0`.
- **iGPU** — Intel Graphics (Core Ultra 9 285K, Arrow Lake), `renderD128`,
  `level_zero:1`, device id `0x7d67`, BusID `PCI:0:2:0`.
- CPU: Intel Core Ultra 9 285K (8 P-cores + 16 E-cores).
- Drivers: `xe` (B70) + `i915` (iGPU). CR default **26.18** (see CR file).
- Both GPUs visible in `sycl-ls`.

### B70 compute throughput (measured)
- **FP64: ~1335 GFLOPS** + **530 GB/s** memory bandwidth — our own clpeak,
  posted in [Ginkgo #2013](https://github.com/ginkgo-project/ginkgo/issues/2013)
  (93% / 87% of internally-computed spec).
- **FP64:FP32 ≈ 1:8** — **strong for a consumer/pro GPU** (consumer NVIDIA is
  ~1:64; Intel Alchemist had *no* native FP64, emulation only). Battlemage FP64
  is **native but rate-limited on the general-purpose XVE units** (not the slow
  Alchemist software-emulation path); `cl_khr_fp64` is exposed natively (no
  emulation flags). FP64-CFD is viable on the B70.
- Data-center contrast: Intel Max/Ponte Vecchio is FP64-first (1:1, ~52 TFLOPS).
- **Consequence:** FP64 throughput is **not** the bottleneck for the (bandwidth-
  bound) pressure solve. (clpeak not installed locally — build from source for a
  fresh number if needed.)

## iGPU-PRIME desktop passthrough (set up for FluidX3D, removed 2026-06-17)

Originally the desktop was forced onto the iGPU to free B70 VRAM for CFD.
That was **four** coupled pieces:

| File | Role |
|---|---|
| `/etc/default/grub` → `desktop_gpu=igpu` kernel flag | toggle |
| `/etc/grub.d/11_desktop_b70` | extra "Desktop auf B70" menu entry (`desktop_gpu=b70`) |
| `/etc/udev/rules.d/61-mutter-primary-igpu.rules` | tags iGPU as mutter preferred primary when `desktop_gpu=igpu` |
| `/etc/X11/xorg.conf.d/20-igpu-display.conf` | forces X11 display onto iGPU (PCI:0:2:0) |

**Removed** because the iGPU passthrough caused problems for the OpenFOAM
work. `scripts/disable-igpu-prime.sh` backs up all four to
`/root/igpu-prime-backup-<ts>/` (+ `restore.sh`), strips `desktop_gpu=igpu`,
deletes the other three, sets `GRUB_TIMEOUT=3`, runs `update-grub`.
Kept: `intel_iommu=igfx_off` (fixes DMA-remap / GEM-BO leak) and
`transparent_hugepage=always`.

**Consequence:** the desktop now uses whichever GPU the **monitor cable**
is plugged into. Monitor on the B70 → desktop runs on B70 and consumes
**~1.15 GB VRAM** (gnome-shell ~0.98 GB + Xwayland ~0.17 GB), reducing
solver headroom. Monitor on the iGPU outputs → B70 VRAM stays free.
(Monitor re-plugging is not an available option — find software paths.)

## OpenFOAM meshing gotcha (cost us a wasted decomposePar)

At OpenFOAM `debug` level 2, `decomposePar`/solvers scan the case dir and
**abort fatally on any filename with spaces** (`fileName::stripInvalid ...
this is considered fatal`). Stray PDFs/PNGs/XLSX with spaces in the case
root crash meshing. **Fix:** move all non-OpenFOAM files (anything with
spaces or office/image extensions) out of the case root before meshing.

## Parallel meshing recipe (OF13, used for the half-res case)

```
blockMesh                                  # background mesh
# (feature edges vehicle.eMesh already present / surfaceFeatures if not)
decomposePar -force                        # to N subdomains
mpirun -np N snappyHexMesh -parallel -overwrite
reconstructPar -constant                   # NB: reconstructParMesh is deprecated in OF13
rm -rf processor*; decomposePar -force     # re-decompose mesh + fields
checkMesh -constant
```
Half-res mesh: blockMesh `(120 60 40)` → `(60 30 20)` (2× coarser/dir) →
snappy → **7.1M cells** (layers nearly double the snapped 3.9M).
