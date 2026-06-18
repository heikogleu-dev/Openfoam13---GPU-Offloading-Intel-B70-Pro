# Config pitfalls — mistakes we hit, so they never cost time again

Hard-won. Each entry: the mistake → the symptom → the fix. Check this before every
run. Pairs with [ogl-ginkgo-config-reference.md](ogl-ginkgo-config-reference.md)
(authoritative flag list) and the copy-paste templates in [`../configs/`](../configs/).

## fvSolution / case setup
- **Wrong keyword silently ignored — `splitComm` vs `splitMPIComm`.** OGL's README
  says `splitComm` but the source key is **`splitMPIComm`** (ExecutorHandler.hpp:274,
  default `true`). Finding 16's `splitComm false` test was a **no-op** — it toggled
  nothing. Always grep the OGL source for the exact key before trusting a test.
- **`regenerate` is NOT a real OGL key.** Re-copy is governed by `updateRHS` /
  `updateInitGuess` / `updateSysMatrix`; preconditioner reuse by `caching` /
  `preconditionerCaching`. Setting `regenerate` does nothing.
- **Heredoc brace mismatch → `foamRun` parse abort.** When generating `fvSolution`
  via a shell heredoc, an extra/missing `}` (e.g. after the `omega` smoother block)
  makes OpenFOAM abort with a dictionary parse error and the GPU is never touched.
  Count braces; prefer copying a known-good template from `configs/` over hand-editing.
- **Forgot `-solver incompressibleFluid` on `foamRun`** → "solver not specified",
  run exits immediately, GPU untouched (looks like a GPU failure but isn't). The
  `-parallel` mpirun line must include the solver flag.
- **Stale README defaults.** OGL README lists defaults that don't match source:
  real defaults are `relaxationFactor 0.6` (not 0.8), `caching 0`, `maxLevels 20`
  (not 9), `minCoarseRows 64000` (not 10). Trust the source, not the README.

## Preconditioner / precision
- **`precision` works only where we patched it.** Upstream OGL exposes `precision`
  on **BJ only**; our `precision double|mixed|single` on **Multigrid** is our patch
  (Preconditioner.hpp, [ginkgo-ogl-stack.md](ginkgo-ogl-stack.md)). On an unpatched
  build the keyword is ignored for MG.
- **`caching > 0` is a dead path without the UpdateMatrixValue extension.** The
  cache-hit branch is `#ifdef GINKGO_WITH_OGL_EXTENSION` (note the singular typo;
  the CMake define is plural `...EXTENSIONS`) and calls `gko::UpdateMatrixValue`,
  which is **absent from Ginkgo 2.0** → it compiles out, so `caching` just full-
  rebuilds anyway. (Fixing this is the AMG-reuse project — [amg-reuse-port-plan.md](amg-reuse-port-plan.md).)
- **`caching` with FULL hierarchy reuse diverges for Navier-Stokes.** Empirically
  `caching 2` (within-step full reuse) blew iterations to the 201 cap — matches the
  literature (Demidov: full reuse is hit-or-miss for N-S). Only *values-only*
  Galerkin refill is safe; keep `caching 0` until the reuse port lands.
- **`GINKGO_FORCE_GPU_AWARE_MPI` is a no-op with `forceHostBuffer true`.** Measured:
  setting it to 0 changed nothing (host buffers are used regardless). Don't burn a
  run testing it in the forceHostBuffer config.

## Build / driver
- **Multi-rank needs the CR 26.05 LD-switch.** CR ≥26.14 aborts `zeInit` for ≥2
  processes (#922). Always `source scripts/cr2605-shell.sh` before an mpirun > 1
  rank. ([intel-compute-runtime-and-driver.md](intel-compute-runtime-and-driver.md))
- **`decomposePar` aborts on filenames with spaces** at OpenFOAM debug level 2
  (stray PDFs/PNGs in the case dir). Move stray files out (`_attachments/`) before
  decomposing.

## Method
- **Same-session A/B only for perf.** GPU clock is sensitive ±2–3%; never compare
  s/step across sessions. Run baseline + variant back-to-back.
- **Per-phase timing before optimizing.** The OGL `verbose 2` timers (init_precond /
  solve / call_update / call_init / copy_x_back) tell you whether the cost is setup
  or solve — we found 72% setup, which redirected effort. Always look before patching.
