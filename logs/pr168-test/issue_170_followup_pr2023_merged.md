## Follow-up — Ginkgo PR #2023 merged 2026-06-02, motivation for PR #168 strengthened

Update on the Ginkgo side: [ginkgo-project/ginkgo#2023](https://github.com/ginkgo-project/ginkgo/pull/2023)
by @nbeams — "Add triangular solver support for dpcpp" via oneMKL trsm —
merged into Ginkgo `develop` on 2026-06-02 14:48 UTC. This closes the
`dpcpp/solver/lower_trs_kernels.dp.cpp:43: generate is not implemented`
gap that previously blocked every IC/ILU-family preconditioner apply on
the SYCL backend.

That makes the OGL → Ginkgo 2.0 migration in PR #168 the only thing
still standing between us and the **first usable strong SYCL
preconditioner** on Battlemage hardware. Specifically, our PR #168 test
on BMG-G31 (write-up at
[finding 23](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/23_pr168_ginkgo_2.0_migration_test.md))
shows the migration still blocks on:

- `include/OGL/Preconditioner.hpp:270` —
  `gko::preconditioner::Ilu<ir, ir>::build()` uses the Ginkgo 1.x
  template form (type parameters); Ginkgo 2.0 changed `Ilu`'s
  `L_solver_type`, `U_solver_type` slots to a non-type parameter. Error
  propagates through 9 TUs, blocks OGL link entirely.

- `src/MatrixWrapper/HostMatrix.cpp:239` — `patch.nbrPatchID()`
  (renamed `nbrPatchIndex()` in OF Foundation 13). One-liner.

If it's helpful, I'm happy to:

1. PR the `nbrPatchIndex()` rename (trivial)
2. Test the Ginkgo 2.0 `Ilu` template migration locally and PR a patch
   into #168 — but would defer to you on the migration semantics
   (whether the non-type parameter should default to something, what
   the new builder pattern looks like in OGL context)
3. Re-run the full preconditioner sweep (BJ blocked, ICT, ILU, Multigrid,
   hybrid format) the moment a buildable OGL ↔ Ginkgo 2.0 lands, and post
   results back here

Either way — wanted to flag that the upstream blocker named in our
finding 05 is now gone, which makes the PR #168 work meaningfully more
valuable on Intel SYCL hardware.

— Heiko
