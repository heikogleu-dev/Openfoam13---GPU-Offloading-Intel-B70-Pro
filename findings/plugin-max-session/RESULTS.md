# Plugin-Maximum session — OGL/Ginkgo pressure solve, 17.2M mixed (2026-06-26)

**Goal:** push the OGL/Ginkgo GPU pressure solve past the measured 1.18× toward the
plugin ceiling via cheap levers (D2H transfer, mesh renumber, decomposition), against a
stable mixed-precision baseline. **NOT full-float** (that's VRAM-only, see full-float-port-plan).

**Setup:** 17.2M (Testcase-mid, 17.22M cells), mixed lib (C-working: double outer + FP32
MG precond via `precision single`), AMG-reuse (Plan C `update_matrix_value`) active,
CR 26.05 LD-switch, np=16, caching 2, 12 steps. Chosen 17.2M over the plan's 28M (no 28M
mesh exists; 17.2M is the ready proven mixed baseline). AMG-reuse (plan's Phase-3 capstone)
was already done+installed → verified, not re-ported.

## Results

| Phase | Lever | s/step | vs Baseline | vs CPU-GAMG (22.1) |
|---|---|---|---|---|
| **0** | Baseline (mixed, reuse on) | **18.73** | 1.00× | **1.18×** |
| 1a | + copy-offload-off (`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`) | 19.58 | **0.96× (worse)** | 1.13× |
| 1b | + pinned host memory | *skipped* (≤2% on H2D fraction of all2all) | — | — |
| 2a | + renumberMesh (RCM, band −22.5×) | 18.82 | **1.00× (nil)** | 1.17× |
| 2b | + decomposition tuning | *skipped* (scotch already 0.77% inter-rank faces) | — | — |
| 3 | AMG-reuse | already in baseline (~1.12× vs no-reuse: 21.0→18.73) | — | — |

Baseline per-solve breakdown (steady, ms): solve_multi_gpu 2386 = init_precond 927
(build 1726 / **reuse 557** ← extension active, 3.1× cheaper refill) + solve 935 +
**all2all 428** + copy_x_back 19 + reorder 17. CCS 29.4%, BCS 9.5%, VRAM 17.1 GB, iters 12.

## Why the cheap levers give ~0 (measured, not assumed)
- **Transfer (Phase 1):** the copy engine is already used (BCS 9.5%); disabling copy-offload
  is −4.5% (BCS→0% confirms the env-var propagated). Real D2H (`copy_x_back`) is 19 ms =
  0.8% of the solve — not a bottleneck. The big transfer is `all2all` 428 ms = MPI gather
  to owner (ranksPerGPU 16→1) + H2D, MPI-repartition-dominated → pinned host memory would
  only touch the H2D fraction (≤2%).
- **Cache locality (Phase 2a):** renumberMesh cut the matrix band **22.5×** (1,075,700→47,765)
  yet s/step didn't move (18.82 vs 18.73) and the SpMV timer even rose → **the CPU side is
  NOT cache-locality-bound**, and GPU SpMV on B70 doesn't benefit from the cell reorder.
- **Partition (Phase 2b):** scotch already yields **0.77%** inter-rank faces (398,408 of
  51.9M) → no halo headroom; renumber (a related locality lever) already gave 0.
- **AMG-reuse (Phase 3):** already in the 1.18× baseline (build 1726 vs reuse 557 ms proves
  it active; Plan C step contribution ~1.12×).

## Conclusion
**The OGL/Ginkgo plugin is at ~1.18× — its practical maximum for the current architecture.**
The plan's 1.45–1.5× target is **not reachable with tuning levers**. The bottleneck is
(a) ~64% CPU-rest (U/k/omega + FVM assembly + flux + MPI) the GPU can't touch and which is
neither cache- nor halo-bound, and (b) the algorithmic GPU-AMG cost (superlinear build,
bandwidth-bound SpMV ≈ roofline). Moving past 1.18× needs **architectural** changes, all
out of this session's scope:
- **GPU-resident assembly** (→ NeoN) — removes the CPU-rest.
- **GPU-aware MPI** (Intel #922 — now fixed in CR 26.22, re-open as a lever).
- **Better AMG** — RS-coarsening / Chebyshev smoother (Ginkgo-side, no SYCL kernels yet).

## Adopted as standard
- **renumberMesh (RCM)** after decomposePar — wired into `sweep-fullfloat.sh`. 22.5× band
  reduction (cache hygiene); no measured B70 speedup here but standard best practice + may
  help other solvers/configs. Harmless.

## State after session
- 17.2M case: renumbered (RCM), np16 decomposed, mixed config (precision single, caching 2).
- Installed libOGL: **C-working (mixed/double)** — the stable baseline. Float (Plan D) is
  one `cp libOGL.so.D-fullfloat-20260619` away.
- CR 26.05 LD-switch (26.22 available, fixes #922 — see intel-compute-runtime-and-driver.md).
