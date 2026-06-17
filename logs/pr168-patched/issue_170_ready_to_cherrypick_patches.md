## Follow-up — Two minimal patches make PR #168 build and run BJ(1) on OF Foundation 13 + Ginkgo 2.0 + Intel BMG-G31

Picking up from our [PR #2023 follow-up comment](#) on the Ginkgo
triangular-solver merge: we applied two minimal patches to your PR #168
branch and got a clean build + working BJ(1) baseline + ILU reaching
the generate phase on Intel Arc Pro B70 Pro (Battlemage G31). Full
write-up: [finding 24](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/24_pr168_patched_ilu_first_test.md).

Both patches are 1-line and ready to cherry-pick onto PR #168.

### Patch 1 — OF Foundation 13 API rename (`HostMatrix.cpp`)

```diff
diff --git a/src/MatrixWrapper/HostMatrix.cpp b/src/MatrixWrapper/HostMatrix.cpp
@@ -236,7 +236,7 @@ std::shared_ptr<SparsityPattern> HostMatrixWrapper::compute_interface_sparsity(
 #ifdef WITH_ESI_VERSION
             const label neighbPatchId = patch.neighbPatchID();
 #else
-            const label neighbPatchId = patch.nbrPatchID();
+            const label neighbPatchId = patch.nbrPatchIndex();
 #endif
```

`cyclicFvPatch::nbrPatchID()` was renamed `nbrPatchIndex()` in
OF Foundation 13 (`finiteVolume/lnInclude/cyclicFvPatch.H:97`):

```cpp
virtual label nbrPatchIndex() const
{
    return cyclicPolyPatch_.nbrPatchIndex();
}
```

### Patch 2 — Ginkgo 2.0 `Ilu` template-form migration (`Preconditioner.hpp`)

```diff
diff --git a/include/OGL/Preconditioner.hpp b/include/OGL/Preconditioner.hpp
@@ -267,7 +267,7 @@ public:
             auto precond_factory =
-                gko::preconditioner::Ilu<ir, ir>::build()
+                gko::preconditioner::Ilu<scalar, false, label>::build()
                     .with_l_solver(gko::clone(trisolve_factory))
                     .with_u_solver(gko::clone(trisolve_factory))
                     .on(device_exec);
```

Ginkgo 2.0 changed `Ilu` template signature from
`Ilu<L_solver_type, U_solver_type, ...>` to
`Ilu<ValueType, ReverseApply, IndexType>`. The runtime `with_l_solver`/
`with_u_solver` builder calls still work; only the template head
changed. With OF's `scalar=double` and `label=int32`, this is equivalent
to `Ilu<>::build()` with defaults — kept explicit for clarity.

### Results with both patches applied

- Configure + build of OGL against Ginkgo `develop` (2.0.0 via CPM):
  clean, all 7 `libginkgo*.so.2.0.0` produced
- BJ(maxBlockSize=1) smoke run: 201 iter, identical residual pattern
  to the Ginkgo 1.11 baseline
- `ILU` reaches `Preconditioner.hpp:219: Generate preconditioner ILU`
  (was `NotImplemented` immediately in 1.x — confirming
  ginkgo-project/ginkgo#2023's `lower_trs` merge unblocks the code path)

ILU then fails with VRAM OOM during `Csr::convert_to(Coo)` setup on
our 34M-cell case (~26.5 GB stable plateau then spike past the 32 GB
ceiling — hardware tightness rather than software bug, full analysis
in finding 24).

### Build-environment adjustments (not patches, just notes)

These are not OGL bugs — drift between PR #168 and the current Intel
stack — but pioneers reproducing this need to know:

1. `cmake --preset release` (not `debug`): with `-O0` the icpx 2026
   `sycl-post-link` step grows past 38 GB RSS on the ~120 SYCL kernels
2. `-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON`: required when a system
   Ginkgo is present; otherwise `find_package(Ginkgo QUIET)` succeeds,
   `install(TARGETS ginkgo ...)` then errors because the target is
   imported rather than a normal target
3. Remove `target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)`
   from `_deps/ginkgo-src/dpcpp/CMakeLists.txt:149` — flag was removed
   in icpx 2026 (now implicit default)
4. Add `${CMAKE_CURRENT_SOURCE_DIR}/foam-shim` to `OGL_public_api`
   `INTERFACE_INCLUDE_DIRECTORIES` — OGL ships `include/foam-shim/fvCFD.H`
   as an OF Foundation 13 shim (`fvCFD.H` was removed in OF13) but the
   include path isn't propagated; trivial 3-line `set_property` addition

Items 2 and 3 are arguably worth fixing in PR #168 itself.

### Note on multi-rank testing

Heads-up for anyone re-testing on Intel BMG: as of Compute Runtime
26.18 (the current release at time of writing), multi-rank OGL hits a
`pthread_once`/`zeInit` race during SYCL platform init that prevents
the solver from reaching the first solve. Full reproducer at
[finding 25](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/25_cr_26.18_multirank_pthread_race.md).
Same class of issue as the previously reported CR 26.14 / `resource_info.cpp:15`
abort. Our successful BJ(1)+ILU smoke tests above were on CR 26.05
(the previously pinned version).

Happy to PR either or both patches upstream against PR #168 directly
if it helps, or you can cherry-pick from the diffs above.

— Heiko
