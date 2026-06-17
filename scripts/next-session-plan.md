# Next-session plan (prepared 2026-06-17 evening)

## Status snapshot
- **Best GPU pressure config found:** Ginkgo **Multigrid, V-cycle, Jacobi
  smoother, CG coarse-solver** (maxIterCoarse 20) → ~13 iters, **~9 s/step**
  at 7.1M = **~1.17× CPU GAMG (7.7 s)**. Set as `Testcase-half` default.
- **GPU is under-fed at 7.1M:** compute engine only ~46% busy, copy ~30%
  (CPU-assembly/transfer bound). → the open question is whether feeding it
  more cells closes the GAMG gap.
- **VRAM ceiling ≈ 20M** cells for MG (double, np=8); ~1.5 GB/M.
- KB written under `knowledge/` (iron-rule maintained). Findings in
  `findings/30`. Everything committed + pushed.

## STEP 1 (tomorrow first): 18M-mesh test — does feeding the GPU help?

Case **already prepared**: `~/CFD-Cases/Testcase-mid` (copied from original
Testcase, blockMesh **90×45×30** = arithmetic mid of half 60×30×20 and
original 120×60×40; best MG config + GPU controlDict already set; STL in
place; stray files moved to `_attachments/`; **not yet meshed**). Predicted
**~18M cells** (range 16–22M; confirm after meshing).

Mesh on **16 cores** (i9 285K handles it), then run on 8 ranks (ranksPerGPU 8):
```bash
cd ~/CFD-Cases/Testcase-mid
blockMesh                                               # ~121k base cells
sed -i -E 's/numberOfSubdomains [0-9]+;/numberOfSubdomains 16;/' system/decomposeParDict
decomposePar -force
mpirun -np 16 snappyHexMesh -parallel -overwrite        # 16-core mesh
reconstructPar -constant                                 # NB reconstructParMesh is deprecated in OF13
rm -rf processor*
sed -i -E 's/numberOfSubdomains [0-9]+;/numberOfSubdomains 8;/' system/decomposeParDict
decomposePar -force                                      # 8-way for the GPU solve
checkMesh -constant                                      # confirm cell count + quality
```
Then the comparison (measure ALL three: iters, s/step, **GPU util**, VRAM):
```bash
source ~/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh
# GPU Multigrid (best config already in system/fvSolution):
mpirun -np 8 foamRun -parallel -solver incompressibleFluid > log.MG-18M 2>&1
# GPU util during it: bash ~/gpu-diag/gpu-util-v2.sh (point CASE at Testcase-mid)
# CPU GAMG baseline (swap p-solver to GAMG, np=8 or np=16):
#   cp system/fvSolution.GAMG-bak system/fvSolution  (or write a GAMG p-block)
```
**Key questions to answer:**
1. Does **compute util rise above 46%** at 18M (GPU better fed)?
2. Does the **MG-vs-GAMG wall-clock gap close** (toward parity/win)?
3. Actual **VRAM at 18M** — confirm it's ~27 GB (under the ~20M... note: 18M
   may sit right at/over the estimate; watch for OOM, fall back to W-cycle
   config which used less VRAM).

## STEP 2 (only if we then want meshes >20M): mixed-precision OGL patch

Rationale: raises VRAM ceiling ~20M → ~25M (DP-SP) / ~30M (aggressive).
**Note from the GPU-util finding:** payoff is VRAM, not speed (compute not the
bottleneck; H→D matrix stays double). So only worth it for >20M meshes.

Implementation (OGL `include/OGL/Preconditioner.hpp`, Multigrid `Schwarz`
branch ~line 469; back it up first):
- Add aliases: `fpgm = gko::multigrid::Pgm<float,label>`, `fir =
  gko::solver::Ir<float>` (fbj/fcg already exist).
- Add `precision` lookup to the Multigrid block (currently only BJ has it,
  line 172): `double` (current) | `mixed` (DP-SP) | `single` (aggressive).
- For `mixed`: build a **heterogeneous** multigrid per Ginkgo's
  `mixed-multigrid-preconditioned-solver.cpp` example —
  `.with_mg_level(pgm_double, fpgm_float)` (level0 double, levels1+ float),
  `.with_pre_smoother(sm_double, sm_float)`, `.with_coarsest_solver(coarse_float)`.
  Restrict mixed mode to Jacobi smoother + CG coarse to keep it tractable.
- Rebuild: `cd /opt/ogl-src && cmake --build build/release` (release preset;
  -O0 debug blows past 38 GB RSS), then copy libOGL.so into
  `$FOAM_USER_LIBBIN`.
- Risk: Ginkgo float Pgm + cross-precision restriction kernels must be
  instantiated in the SYCL build; distributed-Schwarz + mixed interaction
  untested. Validate with a 5-step sanity run before the 100-step smoke.
- Test on `Testcase-half` (7.1M): **safe DP-SP** 100-step smoke (iters,
  s/step, VRAM, stability) → then **aggressive (all-float precond)**.
- Reference configs: `knowledge/gpu-amg-reference-configs.md`.

## Also worth trying (cheap, available now)
- **W-cycle** MG config (~15 iters, 12 s/step, **10.4 GB** — leaner VRAM than
  CG-coarse's 11.5 GB) as a fallback if 18M is VRAM-tight.
- **GPU-aware MPI** (drop `forceHostBuffer`) — would attack the ~30% copy
  engine; needs an MPI built with Level-Zero support (check availability).

## Don't forget
- Meshing: filenames with spaces in the case dir crash decomposePar at OF
  debug level 2 — keep the case root clean (stray files already in
  `_attachments/`).
- Multi-rank needs the CR 26.05 LD-switch (`scripts/cr2605-shell.sh`).
