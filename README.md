# Intel Arc Pro B70 + OpenFOAM CFD — Pioneer Documentation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-Foundation%2013-green)](https://openfoam.org)
[![Ginkgo](https://img.shields.io/badge/Ginkgo-2.0%20SYCL-orange)](https://github.com/ginkgo-project/ginkgo)
[![Intel Arc Pro B70](https://img.shields.io/badge/Intel%20Arc%20Pro-B70%2032GB-blue)](https://www.intel.com/content/www/us/en/products/sku/245797)
[![GPU CFD Status](https://img.shields.io/badge/GPU%20pressure%20solve-BEATS%20CPU%20GAMG-brightgreen)]()

> **TL;DR (June 2026): the GPU now wins.** With Ginkgo 2.0 + a tuned algebraic
> **Multigrid** preconditioner + **single-precision** preconditioning (our OGL
> patch), the B70 GPU pressure solve **beats CPU GAMG** on this hardware — the
> first clear GPU-CFD pressure-solve win documented on Intel Battlemage.
>
> | 17.2M cells, np=8/16 | s/step | iters |
> |---|---|---|
> | CPU GAMG (np=16) | 22.1 | 5–9 |
> | GPU Multigrid, double (np=16) | 22.7 | ~13 |
> | GPU Multigrid, single precision (np=16) | 20.9 | ~13 |
> | **GPU MG, single + AMG-reuse (np=16)** | **18.67** ✅ ~1.18× | ~13 |
>
> The May-2026 verdict ("software not ready, CPU wins 1.5×") is kept below as the
> historical record — it was true for the Ginkgo-1.11/BJ(1) stack. Two months of
> work (Ginkgo 2.0 migration, Multigrid tuning, a mixed-precision OGL patch)
> turned the result around.

First public documentation of OpenFOAM GPU acceleration (OGL/Ginkgo/SYCL) on
**Intel Arc Pro B70 (Battlemage)** for automotive CFD. Honest results, real bugs
found and reported upstream, every config and finding reproducible.

---

## The performance journey (why it took until June to win)

GPU-accelerating an **implicit FVM pressure (Poisson) solve** is one of the
hardest things to put on a GPU — even NVIDIA's mature AmgX often only ties CPU
GAMG. The bottleneck is never raw compute; it's the **preconditioner's
convergence rate** and the **communication** around the solve.

| Preconditioner (our path) | 7.1M s/step | 17.2M s/step | why |
|---|---|---|---|
| Block-Jacobi(1) | never converges (201-iter cap) | — | ≈ diagonal scaling, too weak |
| Block-Jacobi(>1) | aborts | aborts | `find_blocks` size_t underflow (OGL distributed) |
| ILU(0) | ~24 (160–201 iter) | ~60 | local precond, ~N^(1/3) iters; OOMs > ~25M |
| **Multigrid (V + CG-coarse), double** | ~9.0 | ~22.7 | converges ~13 iters; **ties CPU GAMG** |
| **Multigrid, single precision** ⭐ | **~7.95** | **~20.9** | **beats CPU GAMG**, −25% VRAM |
| *CPU GAMG (reference)* | 8.5–9.3 | 22.1–25.9 | optimal multigrid, hierarchy cached |

Full measured maps (preconditioner × ranks 2–16 × {util, VRAM, iters,
wall-clock}) in [`knowledge/performance-maps.md`](knowledge/performance-maps.md).

**Why single precision wins:** the SpMV that dominates the solve is
memory-bandwidth-bound; FP32 halves the bytes (≈2× effective bandwidth) and the
VRAM, with **no accuracy/iteration penalty at the solver tolerances used**
(same ~13 iterations as double). B70 FP64 is *not* the bottleneck — it is
respectable (~1.3 TFLOPS, ~1:8, our clpeak in Ginkgo #2013).

---

## The #1 remaining lever (full diagnostic)

A per-phase breakdown (OGL's own `TIME_WITH_FIELDNAME` timers + Ginkgo
ProfilerHook) found where the GPU time actually goes per pressure solve:

| phase | share of GPU pressure-solve | note |
|---|---|---|
| **init_precond (AMG hierarchy rebuild)** | **~50–59%** | **rebuilt EVERY solve** |
| solve (CG loop, MG-apply 96% of it) | ~40% | coarse-CG ≈50% of the apply; SpMV dominant |
| call_update (H2D matrix values) | small | |
| copy_x_back (D2H) | tiny | 0.94 GB/s — a *software transfer-path* issue (pageable host mem / tiny copies), **not** the PCIe link (see note below) |

**The AMG hierarchy is rebuilt on every solve — OpenFOAM GAMG avoids this
(`cacheAgglomeration`).** Caveat (audited): the GPU pressure-solve is only ~40–48%
of the wall-clock step (CPU U/k/omega + assembly is the rest), so eliminating the
rebuild is worth **~10–20% wall-clock**, not 2× — but it's the biggest single
GPU-side lever and puts us ahead of the field (Ginkgo has no reuse API). The fix
is identified: OGL has a
`caching` mechanism, but the value-update path (`gko::UpdateMatrixValue`) is an
hpsim Ginkgo-1.11-fork extension that was lost in the Ginkgo-2.0 migration.
Porting it back (+ a Ginkgo SYCL rebuild) is the top roadmap item — see
[`knowledge/per-iteration-diagnostics.md`](knowledge/per-iteration-diagnostics.md).

---

## 📚 Knowledge Base (start here)

Everything we've tested/measured/researched, cross-checked against external
literature, lives in [`knowledge/`](knowledge/) and is maintained as an iron
rule (search it first, cite external validation):

| Topic | |
|---|---|
| [preconditioners-and-gpu-cfd.md](knowledge/preconditioners-and-gpu-cfd.md) | Why ILU/BJ lose, why AMG wins; the Multigrid tuning map; theory + external validation |
| [performance-maps.md](knowledge/performance-maps.md) | Measured (preconditioner × ranks) maps for 7.1M + 17.2M |
| [per-iteration-diagnostics.md](knowledge/per-iteration-diagnostics.md) | Per-phase breakdown; the AMG-rebuild lever; the fix path |
| [amg-reuse-port-plan.md](knowledge/amg-reuse-port-plan.md) | The #1 lever — port plan for AMG values-only reuse |
| [full-float-port-plan.md](knowledge/full-float-port-plan.md) | Plan D — full-float solve (VRAM lever for 30–35M) |
| [config-pitfalls.md](knowledge/config-pitfalls.md) | Config mistakes we hit — check before every run |
| [intel-b70-tuning-levers.md](knowledge/intel-b70-tuning-levers.md) | Real-vs-no-op triage of Intel/L0/SYCL/xe tuning flags; TOP-5 cheap levers |
| [vram-and-mesh-scaling.md](knowledge/vram-and-mesh-scaling.md) | Per-preconditioner VRAM, ceilings, mixed-precision savings |
| [gpu-amg-reference-configs.md](knowledge/gpu-amg-reference-configs.md) | Proven AmgX / Hypre / Ginkgo pressure configs from the literature |
| [intel-platform-fit.md](knowledge/intel-platform-fit.md) | Are we on the right track on Intel? FP64 reality, community fit |
| [ginkgo-ogl-stack.md](knowledge/ginkgo-ogl-stack.md) | SYCL preconditioner bugs (fixed in 2.0), OGL patches, mixed-precision patch |
| [intel-compute-runtime-and-driver.md](knowledge/intel-compute-runtime-and-driver.md) | CR 26.05/26.18, the multi-process zeInit abort, the LD-switch |
| [hardware-system-grub.md](knowledge/hardware-system-grub.md) | B70 specs (incl. measured FP64), GRUB/desktop-GPU |
| [ogl-ginkgo-config-reference.md](knowledge/ogl-ginkgo-config-reference.md) | **Config reference** — every OGL/Ginkgo flag, valid values, defaults, B70 recommendation |
| [gpu-comparison.md](knowledge/gpu-comparison.md) | B70 vs current AMD/NVIDIA/Intel GPUs (bandwidth-bound CFD); VRAM-per-$ verdict |
| [external-references.md](knowledge/external-references.md) | Papers + projects — the prior art we build on |

---

## Part of the Battlemage CFD Pioneer Series

1. **[FluidX3D-Intel-B70](https://github.com/heikogleu-dev/FluidX3D-Intel-B70)** — LBM via OpenCL. ~6750 MLUPS (production-ready vehicle-aero sandbox; FP32/FP16, GPU-native).
2. **This repo** — implicit FVM pressure solver via Ginkgo SYCL. **GPU now beats CPU GAMG with the right config.**
3. **[Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70)** — PETSc-Kokkos-SYCL attempt; documents what doesn't work yet.

LBM (FluidX3D) is the GPU-native best case; implicit FVM (this repo) is the hard
case where GPUs struggle everywhere — and where we now win on Battlemage.

---

## Hardware Performance — B70 is Capable

| Metric | Measured | Note |
|---|---|---|
| **FP64 compute** | **~1.3 TFLOPS** (~1:8 of FP32) | clpeak; strong for a consumer/pro GPU (NVIDIA consumer ~1:64; Intel Alchemist had none). Not the CFD bottleneck. |
| **VRAM bandwidth** | **530 GB/s** (87% of 608 spec) | the metric that matters for bandwidth-bound CFD |
| FP32 compute | ~12–23 TFLOPS | |
| VRAM | 32 GB GDDR6 | fits ~20M cells (double MG), ~25–28M (single) |
| Kernel-launch latency | 5.6 µs sync / 1.5 µs batched | on par with CUDA |
| PCIe link | Gen5-class (~48–56 GB/s, clpeak) | the "Gen1×1" lspci reading is an Arc switch-hierarchy **artifact** (read the parent bridge); our 0.94 GB/s D2H is a software transfer-path issue, not the link |

The two metrics that matter for CFD — VRAM bandwidth and FP64 — are both solid.
The limiter is the **SYCL solver software** (preconditioner setup + communication),
not the silicon.

---

## System & Software Stack (June 2026)

| Component | Version |
|---|---|
| GPU | Intel Arc Pro B70 (Battlemage BMG-G31) — 32 GB GDDR6 |
| CPU | Intel Core Ultra 9 285K (8P+16E) |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-22 |
| OpenFOAM | Foundation 13 |
| **Ginkgo** | **2.0 (develop) SYCL** — the SYCL preconditioner bugs are fixed here |
| **OGL** | PR #168 + our patches: OF13 API fix, Ginkgo-2.0 ILU template, **mixed-precision multigrid (`precision double\|mixed\|single`)** — [`findings/code/ogl-patches/`](findings/code/ogl-patches/) |
| Intel oneAPI | 2026.0 |
| Intel Compute Runtime | system 26.18; **multi-rank via CR 26.05 LD-switch** (no sudo) — CR ≥26.14 aborts multi-process `zeInit` ([#922](https://github.com/intel/compute-runtime/issues/922)) |

**Best working config:** `GKOCG` + `Multigrid` (V-cycle, Jacobi smoother,
`coarseSolver CG`, `maxIterCoarse 20`, **`precision single`**), `ranksPerGPU` =
`mpirun -np` (8 is the saturation sweet spot), under the CR 26.05 LD-switch.
See [`scripts/cr2605-shell.sh`](scripts/cr2605-shell.sh) and the working
fvSolution in `configs/`.

---

## Findings (all new, upstream-reported where applicable)

Findings 01–29 are the pioneer trail (PCIe reporting bug → the 4 Ginkgo SYCL
preconditioner bugs → CR 26.18 root cause → CR 26.05 LD-switch). Highlights:

- [26](findings/26_ginkgo_2.0_standalone_sweep.md) — the four 1.10/1.11 SYCL
  preconditioner bugs (BJ>1 `find_blocks`, ICT `add_candidates`, `lower_trs`,
  Multigrid PGM) are all **fixed in Ginkgo 2.0**.
- [27](findings/27_cr2605_ld_switch_workaround.md) — CR 26.05 LD-switch restores
  multi-rank OGL with no sudo.
- [29](findings/29_cr_26.18_root_cause_pure_l0_multiprocess_abort.md) — **root
  cause** of the multi-rank break: CR ≥26.14 `abort()` in `zeInit` for ≥2
  processes (pure-L0 reproducer); upstream [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922).
- [30](findings/30_post_recovery_clean_multirank_perf.md) — **the turnaround**:
  clean-boot multi-rank perf; BJ(1)/ILU/Multigrid verdict; the Multigrid tuning
  map; mixed-precision; the GPU-beats-GAMG result; full per-phase diagnostic.

(Full table with 01–29 retained in [findings/](findings/).)

---

## Repository Structure

```
├── README.md                — this file
├── conclusions.md           — honest CPU-vs-GPU verdict (June update at top)
├── knowledge/               — ★ the maintained knowledge base (10 topics)
├── findings/                — 30 findings (the pioneer trail)
│   └── code/                — standalone reproducers (gpu-diag/, ogl-patches/)
├── scripts/                 — cr2605-shell.sh, next-session-plan.md, sweeps
├── logs/                    — raw diagnostic logs + the performance maps
├── configs/                 — working fvSolution configurations
└── references.md            — upstream papers + projects
```

---

## Status: June 2026

**The GPU pressure solve beats CPU GAMG** on this hardware with Ginkgo 2.0 +
tuned Multigrid + single precision. For production today, either path is viable;
the GPU frees CPU cores and wins outright at ≥17M cells.

**Where this is headed (PROJECTED — baseline measured, rest projected; 17.2M np16):**

| Stage | s/step | vs CPU GAMG (22.1) |
|---|---|---|
| Baseline today (single-MG) | **20.9** ✅ | 1.06× |
| + C (AMG reuse) | ~16–17 | ~1.3–1.4× |
| + C + D + tuning | ~14–15 | ~1.4–1.6× |
| Hard ceiling (GPU p-solve → 0) | ~11.5 | ~1.9× |

The GPU does only the pressure-solve (~40–48% of wall-clock); the CPU U/k/omega +
assembly is the rest, so **p-only offload tops out at ~1.9× vs GAMG** (Amdahl) —
beyond that needs offloading the momentum/turbulence solves too. The margin **grows
with mesh size** (B70 under-fed below ~10M). Details + uncertainties in
[`knowledge/performance-maps.md`](knowledge/performance-maps.md).

**Top roadmap items** (see [`scripts/next-session-plan.md`](scripts/next-session-plan.md)):
1. **AMG-hierarchy caching with value-update** (port the `update_matrix_value`
   extension to Ginkgo 2.0) — ~2× on the GPU pressure-solve = **~15–20% wall-clock**
   (the GPU p-solve is ~40–48% of the step; CPU U/k/omega + assembly is the rest).
   Biggest GPU-side lever; see [`knowledge/amg-reuse-port-plan.md`](knowledge/amg-reuse-port-plan.md).
2. **Full-float solve** (re-template the OGL solve path) — a **VRAM** lever for the
   34M mesh on a single 32 GB card (not a speed lever).
3. **Cheap tuning A/Bs** (no rebuild): diagnose the 0.94 GB/s D2H with clpeak first
   (the link is fine), SYCL copy-engine / V1-batching, GPU-clock pinning — see
   [`knowledge/intel-b70-tuning-levers.md`](knowledge/intel-b70-tuning-levers.md).
4. Watch: Ginkgo classical Ruge-Stüben AMG (landed in develop, SYCL kernels pending).

---

## How to Cite

```bibtex
@misc{gleu2026battlemage,
  author = {Gleu, Heiko},
  title  = {Intel Arc Pro B70 + OpenFOAM CFD: Pioneer Documentation of GPU
            Acceleration via OGL/Ginkgo/SYCL on Battlemage},
  year   = {2026},
  url    = {https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro}
}
```

## Related Projects
- [hpsim/OGL](https://github.com/hpsim/OGL) · [ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo) · [intel/compute-runtime](https://github.com/intel/compute-runtime)
- [Phoronix B70 Linux Benchmarks](https://www.phoronix.com/review/intel-arc-pro-b70-linux)

## Community
Different results on your hardware? A fix? → [Open an Issue](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/issues) ·
[CFD-Online](https://www.cfd-online.com/Forums/openfoam/) · [r/IntelArc](https://reddit.com/r/IntelArc)

## License
GPL-3.0-or-later — see [LICENSE](LICENSE). Third-party attribution in [NOTICE.md](NOTICE.md).

---

<details>
<summary><b>Historical record — May 2026 verdict (superseded)</b></summary>

The original May-2026 conclusion was **"hardware excellent, software not ready;
CPU GAMG wins by 1.5×"** — true for the Ginkgo-1.11 / BJ(1) stack, where BJ(1)
never converged (201-iter cap) and no strong SYCL preconditioner worked. The
34M-cell automotive case ran CPU GAMG at 35.7 s/step (np=16) vs GPU BJ(1)
~53 s/step. That stack had: no GPU multigrid, IC `NotImplemented`, the BJ>1
`find_blocks` underflow, and GMRES hitting the 32 GB VRAM ceiling. The Ginkgo
2.0 migration + Multigrid tuning + mixed precision (documented above and in
`findings/30`) superseded that verdict. Full May detail in `conclusions.md` and
findings 01–29.

</details>
