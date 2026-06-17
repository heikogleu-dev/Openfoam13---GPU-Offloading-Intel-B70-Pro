# External references — the prior art we build on

Cross-checks for our findings. **Search this + the other KB files before
doing fresh external research.**

## Preconditioner theory (why ILU/BJ lose, why AMG wins)

- **Saad, *Iterative Methods for Sparse Linear Systems* (2nd ed.)** —
  canonical: ILU(0)/Jacobi give a constant-factor κ improvement but keep
  the O(h⁻²) asymptotics → CG iterations ~O(h⁻¹) = O(N^(1/3)) in 3-D;
  multigrid is mesh-independent (O(1) iterations, O(N) work).
- "One-level PCG solvers deteriorate for medium-to-large problems while
  multigrid multi-level PCG is fast" — multiple AMG-PCG papers, e.g.
  [An efficient AMG-PCG solver](https://www.sciencedirect.com/science/article/abs/pii/S004578250200378X),
  [A fast AMG-PCG solver](https://www.sciencedirect.com/science/article/abs/pii/S0096300305010416).
  Validates our 7.1M result (ILU ~192 iters ≈ N^(1/3), GAMG 3–5).

## GPU OpenFOAM pressure-solve (the win = GPU algebraic multigrid)

- **NVIDIA AmgX for OpenFOAM** — GPU AMG preconditioner for the pressure
  PCG; up to **9× pressure-solve speedup** vs GAMG-PCG, ~2–3× overall
  (Amdahl). Confirms: the GPU win comes from **AMG on the GPU**, not ILU/BJ.
  [Martineau/NVIDIA OpenFOAM 2020/2021](https://wiki.openfoam.com/images/a/a4/OpenFOAM_2020_NVIDIA_Martineau.pdf),
  [OpenFOAM with GPU Solver Support](https://www.keysight.com/cae/sites/default/files/resource/other/2611/OpenFOAM_Conference_2021_Abstract_Martineau_NVIDIA.pdf).
- **petsc4Foam** (+ Hypre / AmgX / cuSPARSE) — maintained GPU plug-in path.
  [arXiv:2403.07882](https://arxiv.org/pdf/2403.07882),
  [NEXTFOAM petsc4Foam](https://blog.nextfoam.co.kr/2024/01/10/gpu-accelerated-openfoam-with-petsc4foam/).
- **RapidCFD** — GPU OpenFOAM fork, unmaintained since 2016 (OF 2.3.1).
- **OpenFOAM on GPUs, state of play (2026)** — [Hivenet overview](https://www.hivenet.com/post/openfoam-gpu-state-of-play).

## Ginkgo / OGL (our stack)

- **Olenik et al. 2024, "Towards a platform-portable linear algebra
  backend for OpenFOAM", Meccanica** —
  [doi:10.1007/s11012-024-01806-1](https://link.springer.com/article/10.1007/s11012-024-01806-1).
  OGL design; KIT guidance "2 MPI subdomains per GPU".
- **Olenik et al., "Towards Distributed Linear Solvers on GPUs using
  Ginkgo"** — [KIT/OpenFOAM 2022](https://www.keysight.com/cae/sites/default/files/resource/other/3280/30_Abstract_OpenFOAM_2022_Olenik_KIT.pdf).
- **Ginkgo has AMG / multigrid on Intel GPUs** (W-cycle, mixed precision,
  Max 1550 / Ponte Vecchio) — [Intel: Ginkgo & oneAPI](https://www.intel.com/content/www/us/en/developer/articles/technical/ginkgo-and-oneapi-accelerate-numerical-simulations.html),
  [Tsai et al., Porting Ginkgo to Intel GPUs, arXiv:2103.10116](https://arxiv.org/pdf/2103.10116).
  → This is exactly the GPU-AMG we want; the open question is VRAM-viability
  + effectiveness through OGL's distributed path on Battlemage.
- [hpsim/OGL](https://github.com/hpsim/OGL), [ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo).

## Our own upstream reports

- [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922)
  — CR ≥26.14 multi-process `zeInit` abort on BMG-G31.
- [Ginkgo PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
  — `lower_trs`/`upper_trs` SYCL gap (merged 2026-06-02), unblocks ILU/IC apply.
- OGL status / patches → `../logs/ogl-greole-consolidated-status.md`,
  OGL issue #170.

## Where our work is genuinely new

The fundamentals above are established. Our novel contribution is the
**Intel Arc Pro B70 (Battlemage) + SYCL/Ginkgo/OGL** combination: the CR
driver multi-process bug, the Ginkgo SYCL preconditioner bugs and their
2.0 fixes, the `find_blocks` distributed-path failure, and the first
measured BJ/ILU/ISAI/**Multigrid** numbers on this hardware.
