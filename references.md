# References

## Primary Papers

- **Olenik et al. 2024** — *Towards a platform-portable linear algebra
  backend for OpenFOAM*, Meccanica
  [doi:10.1007/s11012-024-01806-1](https://doi.org/10.1007/s11012-024-01806-1)
  → Defines the OGL design and KIT recommendation
  *"2× MPI subdomains per GPU"*. We tested with `ranksPerGPU 8`
  (single GPU, 8 ranks) per this guidance.

- **Tsai et al. 2023** — *Providing performance portable numerics for
  Intel GPUs*, Wiley CCPE
  [doi:10.1002/cpe.7400](https://doi.org/10.1002/cpe.7400)
  → Documents `ParIC` / `ParILU` / `ParICT` / `ISAI` work on DPC++.
  **Earlier versions of this repo claimed a discrepancy with the paper —
  that was wrong.** Per Ginkgo team feedback (issue #2013), `ParIc/ParIlu`
  factorization *does* work on SYCL. The gap we hit on Battlemage is on
  the *apply* side: `lower_trs` / `upper_trs` kernels are missing in
  `dpcpp/solver/`, and `ParIct::add_candidates` SIGABRTs. The classic
  `Ic`/`Ilu` (sparselib-based) is genuinely not in SYCL. See
  [findings/05](findings/05_sycl_preconditioner_status.md) for the
  corrected mapping.

- **Anzt et al. 2022** — *Ginkgo: A Modern Linear Operator Algebra
  Framework for High Performance Computing*, ACM TOMS
  [doi:10.1145/3480935](https://doi.org/10.1145/3480935)
  → Architecture / executor model that OGL builds on.

## OGL / Ginkgo Upstream

- [hpsim/OGL](https://github.com/hpsim/OGL) — OpenFOAM Ginkgo Layer (GPU plugin)
  - [findings/10 issue body (ready to file)](findings/10_ginkgo2_issue_body.md)
- [ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo)
  - [findings/11 issue body (ready to file)](findings/11_ginkgo_issue_body.md)
- [intel/compute-runtime](https://github.com/intel/compute-runtime)
  - Bug filing planned for findings/13 (resource_info abort with multi-rank OGL)

## Related Battlemage Pioneer Work

- **PMZFX/intel-arc-pro-b70-benchmarks**
  https://github.com/PMZFX/intel-arc-pro-b70-benchmarks
  → Independent B70 Pro pioneer for LLM inference. Upstreamed Q8_0 SYCL
  fix (PRs #21527 / #21638 in `llama.cpp`), achieving 3.1× speedup.
  → Validates our broader observation that Battlemage SYCL kernels need
  targeted fixes per workload — not a generic driver/compiler issue.

- **llama.cpp Issue #21517**
  https://github.com/ggml-org/llama.cpp/issues/21517
  → "Update from CR 26.05 to 26.09 did not improve performance — issue is
  in kernel code, not driver." Same pattern as our
  [findings/13](findings/13_stack_update_zeinit_race.md): driver updates
  alone do not solve the per-workload software-stack problems.

## Phoronix Hardware Reviews

- [Intel Arc Pro B70 Linux Benchmarks (Phoronix)](https://www.phoronix.com/review/intel-arc-pro-b70-linux)
  → Reference benchmarks on the same hardware for non-CFD workloads
  (rendering, video, ML inference). Useful for hardware sanity-check
  comparisons.

## Related Hardware/Software Documentation

- [OGL/Ginkgo recommended fvSolution patterns](https://github.com/hpsim/OGL/blob/dev/README.md)
  → Source for SPD-preconditioner `scaling -1.0` requirement we tested
  in [findings/15](findings/15_scaling_for_spd_preconditioners.md).
- [Intel Compute Runtime release notes](https://github.com/intel/compute-runtime/releases)
- [Ginkgo release notes](https://github.com/ginkgo-project/ginkgo/releases)
- [oneAPI Base Toolkit notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-toolkit-release-notes.html)
