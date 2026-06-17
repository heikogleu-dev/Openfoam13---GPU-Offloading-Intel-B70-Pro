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
| [preconditioners-and-gpu-cfd.md](preconditioners-and-gpu-cfd.md) | **The core finding.** Why ILU/BJ lose to GAMG; the Multigrid tuning map (tuned to ~1.17× GAMG); theory (ILU ~N^(1/3) vs AMG ~O(1)); the PCG-not-GAMG baseline caveat; the AMG-resetup gotcha; cells/GPU win threshold |
| [gpu-amg-reference-configs.md](gpu-amg-reference-configs.md) | Proven AmgX / Hypre-BoomerAMG / Ginkgo pressure configs from the literature — the tuning reference |
| [intel-platform-fit.md](intel-platform-fit.md) | **Are we on the right track on Intel?** B70 FP64 reality (strong, ~1335 GFLOPS measured), community alignment/divergences, what's novel (we're the only ones), upstream roadmap (classical AMG incoming) |
| [ginkgo-ogl-stack.md](ginkgo-ogl-stack.md) | Ginkgo SYCL preconditioner bugs (fixed in 2.0), the `find_blocks` distributed-path bug, OGL build patches + valid preconditioner keywords |
| [intel-compute-runtime-and-driver.md](intel-compute-runtime-and-driver.md) | CR 26.05/26.14/26.18, the multi-process `zeInit` abort, the CR 26.05 LD-switch, GPU device-lost recovery |
| [vram-and-mesh-scaling.md](vram-and-mesh-scaling.md) | Per-preconditioner VRAM (bytes/row), what fits at what mesh size on 32 GB, measured peaks |
| [hardware-system-grub.md](hardware-system-grub.md) | B70 specs, iGPU-PRIME GRUB setup/removal, desktop-on-B70 VRAM cost |
| [external-references.md](external-references.md) | Papers, upstream projects, and links — the prior art we are building on |

## One-line summary of the whole investigation

The B70 hardware and the SYCL/Ginkgo/OGL stack work. Block-Jacobi and ILU
are one-level preconditioners whose iteration count grows with mesh size,
so they lose to CPU GAMG (BJ never converges; ILU ~N^(1/3) iters, ~3×
slower). **The one that works is Ginkgo's algebraic Multigrid, and tuning
it (V-cycle + Jacobi smoother + CG coarse-solver) gets to ~13 iters /
~9 s/step at 7.1M — only ~1.17× CPU GAMG (7.7 s), on an under-fed GPU.**
GPU-AMG is the right path (as NVIDIA AmgX proves). Two structural caveats
keep it from clearly winning yet: Ginkgo has only PGM aggregation (no
classical AMG → floors at ~13 iters, not GAMG's 3–5), and the AMG
hierarchy must be re-setup each timestep (GAMG caches its). The most
promising next step is a larger mesh (~15–18M) to feed the GPU above the
~10M-cells/GPU win threshold.
