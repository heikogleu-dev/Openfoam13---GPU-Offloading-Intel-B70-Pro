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
| [preconditioners-and-gpu-cfd.md](preconditioners-and-gpu-cfd.md) | **The core finding.** Why ILU/BJ lose to GAMG, why GPU-AMG is the only path, theory (ILU(0) ~N^(1/3) vs AMG ~O(1)) + our measurements + external validation (AmgX, Ginkgo MG) |
| [ginkgo-ogl-stack.md](ginkgo-ogl-stack.md) | Ginkgo SYCL preconditioner bugs (fixed in 2.0), the `find_blocks` distributed-path bug, OGL build patches + valid preconditioner keywords |
| [intel-compute-runtime-and-driver.md](intel-compute-runtime-and-driver.md) | CR 26.05/26.14/26.18, the multi-process `zeInit` abort, the CR 26.05 LD-switch, GPU device-lost recovery |
| [vram-and-mesh-scaling.md](vram-and-mesh-scaling.md) | Per-preconditioner VRAM (bytes/row), what fits at what mesh size on 32 GB, measured peaks |
| [hardware-system-grub.md](hardware-system-grub.md) | B70 specs, iGPU-PRIME GRUB setup/removal, desktop-on-B70 VRAM cost |
| [external-references.md](external-references.md) | Papers, upstream projects, and links — the prior art we are building on |

## One-line summary of the whole investigation

The B70 hardware and the SYCL/Ginkgo/OGL stack work. Block-Jacobi and ILU
are one-level preconditioners whose iteration count grows with mesh size,
so they lose to CPU GAMG (BJ never converges; ILU ~N^(1/3) iters, ~3×
slower). **The one that works is Ginkgo's algebraic Multigrid: at 7.1M it
runs (10.4 GB), converges in 55–101 iters, and is ~14 s/step — only ~1.8×
slower than CPU GAMG, and that is the *untuned* default.** GPU-AMG is the
right path (as NVIDIA AmgX proves); the open work is tuning Ginkgo
Multigrid (W-cycle, smoother, levels) and making it VRAM-viable at
production mesh sizes through OGL's distributed path.
