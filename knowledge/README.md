# Knowledge Base — Intel Arc Pro B70 OpenFOAM GPU Offloading

This is the project's **searchable knowledge base**: everything we have
tested, measured, learned, or researched — each topic cross-checked
against external literature so we don't re-discover what the field already
knows.

## 🛑 IRON RULE (how this KB is used)

1. **Search here FIRST.** Before any external web search or fresh
   investigation, grep this `knowledge/` directory. If the answer is here,
   use it (and trust the cross-checked external validation already cited).
2. **Cross-check against external sources.** Every non-trivial finding
   must be validated against published literature / upstream projects, and
   the source linked in the topic file. If our result *contradicts* an
   external source, flag the conflict explicitly — do not silently pick one.
3. **Write findings here immediately.** A new measurement, bug, or
   research result goes into the relevant topic file in the same session it
   was produced — not "later". Commit + push.
4. **Distinguish ours vs known.** Mark each fact as either *established in
   the literature* (with citation) or *our own measurement on this stack*.
   The novel contribution of this project is the **Intel Battlemage +
   SYCL/Ginkgo/OGL** combination, not the linear-algebra fundamentals.

This mirrors the KB discipline used in the sister FluidX3D project.

## Topics

| File | What it covers |
|---|---|
| [preconditioners-and-gpu-cfd.md](preconditioners-and-gpu-cfd.md) | **The core finding.** Why ILU/BJ lose to GAMG; the Multigrid tuning map (single precision **beats** GAMG); theory (ILU ~N^(1/3) vs AMG ~O(1)); the PCG-not-GAMG baseline caveat; the AMG-resetup gotcha; cells/GPU win threshold |
| [gpu-amg-reference-configs.md](gpu-amg-reference-configs.md) | Proven AmgX / Hypre-BoomerAMG / Ginkgo pressure configs from the literature — the tuning reference |
| [per-iteration-diagnostics.md](per-iteration-diagnostics.md) | Per-phase breakdown (init_precond/solve/transfer); the AMG-rebuild lever; fix path |
| [amg-reuse-port-plan.md](amg-reuse-port-plan.md) | The #1 lever — full port plan for AMG values-only reuse (interface, develop adaptation, build/test) |
| [full-float-port-plan.md](full-float-port-plan.md) | Plan D — full-float solve (VRAM lever for 30–35M); scope, contained-conversion approach, memory-accessor fallback |
| [config-pitfalls.md](config-pitfalls.md) | Mistakes we hit (wrong keys, brace errors, dead caching path, no-op flags) — check before every run |
| [performance-maps.md](performance-maps.md) | Measured (preconditioner × ranks) maps for 7.1M + 17.2M: util/VRAM/iters/wall-clock. Verdict: GPU-MG ≈ CPU-GAMG in double; FP32 needed for a clear win |
| [why-no-speedup-at-34M.md](why-no-speedup-at-34M.md) | **Why a busier GPU (36% util) ≠ speedup at 34M.** Step decomposition (GPU-p vs GAMG-p erodes 2.03×→1.66×→0.87×): util is a rate not throughput; GPU-AMG scales superlinear (build 2.85×) vs GAMG sublinear; 64% CPU-rest. Full-float 34M = VRAM win only |
| [intel-platform-fit.md](intel-platform-fit.md) | **Are we on the right track on Intel?** B70 FP64 reality (strong, ~1335 GFLOPS measured), community alignment/divergences, what's novel (we're the only ones), upstream roadmap (classical AMG incoming) |
| [ginkgo-ogl-stack.md](ginkgo-ogl-stack.md) | Ginkgo SYCL preconditioner bugs (fixed in 2.0), the `find_blocks` distributed-path bug, OGL build patches + valid preconditioner keywords |
| [ogl-ginkgo-config-reference.md](ogl-ginkgo-config-reference.md) | **Full cited config reference.** Every fvSolution solver key, all Multigrid sub-options, all preconditioner keywords + options, env vars, build flags — grounded in OGL `dev` source + Ginkgo `develop`, with README-vs-source conflicts + B70 recommendations |
| [intel-compute-runtime-and-driver.md](intel-compute-runtime-and-driver.md) | CR 26.05/26.14/26.18, the multi-process `zeInit` abort, the CR 26.05 LD-switch, GPU device-lost recovery |
| [intel-b70-tuning-levers.md](intel-b70-tuning-levers.md) | Real-vs-no-op triage of Intel/L0/SYCL/xe tuning flags; TOP-5 cheap levers |
| [vram-and-mesh-scaling.md](vram-and-mesh-scaling.md) | Per-preconditioner VRAM (bytes/row), what fits at what mesh size on 32 GB, measured peaks |
| [hardware-system-grub.md](hardware-system-grub.md) | B70 specs (PCIe Gen1×1 = artifact), iGPU-PRIME GRUB setup/removal, desktop-on-B70 VRAM cost |
| [gpu-comparison.md](gpu-comparison.md) | B70 vs AMD/NVIDIA/Intel GPUs (bandwidth-bound CFD); VRAM-per-$ verdict |
| [external-references.md](external-references.md) | Papers, upstream projects, and links — the prior art we are building on |

## One-line summary of the whole investigation

The B70 hardware and the SYCL/Ginkgo/OGL stack work. Block-Jacobi and ILU
are one-level preconditioners whose iteration count grows with mesh size,
so they lose to CPU GAMG (BJ never converges; ILU ~N^(1/3) iters, ~3×
slower). **The one that works is Ginkgo's algebraic Multigrid — tuned
(V-cycle + Jacobi smoother + CG coarse) + single-precision (our OGL patch),
it BEATS CPU GAMG: 17.2M np16 20.9 vs 22.1 s/step, 7.1M 7.95 vs 8.5–9.3,
−25% VRAM, same ~13 iters.** Double precision only ties; the win comes from
FP32 halving the bandwidth-bound SpMV (B70 FP64 is fine, not the wall).
GPU-AMG is the right path (as NVIDIA AmgX proves). **#1 remaining lever:**
the AMG hierarchy is rebuilt every solve (Ginkgo has no reuse API) →
values-only reuse = ~2× on the GPU pressure-solve = ~15–20% wall-clock (the
GPU p-solve is only ~40–48% of the step; the rest is CPU U/k/omega +
assembly) — see [amg-reuse-port-plan.md](amg-reuse-port-plan.md). Structural
note: Ginkgo has only PGM aggregation (RS-coarsening landed in develop but
CPU-only) so it floors at ~13 iters, not GAMG's 3–5 — yet still wins on
per-iteration cost.
