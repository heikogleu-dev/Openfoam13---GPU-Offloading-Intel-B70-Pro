# Intel Compute Runtime & xe driver

## ★★★ RESOLVED in CR 26.22.38646.4 (verified 2026-06-26)
**CR 26.22.38646.4 (+ IGC 2.36.3) fixes the multi-rank `resource_info.cpp:15` abort**
(our issue #922 / GSD-12696). Intel (@ola2308, 2026-06-18) asked us to test it; we did,
via the pure-L0 reproducer `diag-mpi-l0` LD-switched to the 26.22 release artifacts:

| Stack (L0 backend) | np=1 | np=8 |
|---|---|---|
| CR 26.18.38308.1 (system) | abort | abort |
| CR 26.05 (our pin) | OK | OK |
| **CR 26.22.38646.4** | **OK** | **OK** |

`sycl-ls` on the switch: `... B70 ... [1.15.38646+4]`. gmmlib unchanged (22.10.0) → fix is
in NEO `libze_intel_gpu` 26.22, not gmmlib. **Perf unchanged** vs 26.05 (7.1M: 7.08→7.09
s/step, identical iters — IGC 2.36.3/LLVM17 gives no SYCL-kernel speedup; our bottleneck
is algorithmic+bandwidth+CPU-bound, not codegen). **No regression.**
- **Adopt off the 26.05 pin:** (a) LD-switch to `~/intel-cr-26.22` (`scripts/cr2622-shell.sh`,
  proven, no sudo) — note this switches only the L0/SYCL backend, not the OpenCL ICD
  (`/etc/OpenCL/vendors` absolute path); or (b) system-install CR 26.22 + IGC 2.36.3 (sudo,
  also fixes OpenCL ICD). Our workload is L0/SYCL → (a) suffices.
- Setup: `gh release download 26.22.38646.4 -R intel/compute-runtime` (libze-intel-gpu1,
  intel-opencl-icd, intel-ocloc, libigdgmm12) + `v2.36.3 -R intel/intel-graphics-compiler`
  (intel-igc-core-2, intel-igc-opencl-2) → `dpkg-deb -x` into `~/intel-cr-26.22/`.
- **#922 reply drafted** (findings/issue-922-reply-draft.md) — NOT yet posted (Heiko posts).

## The CR 26.14–26.18 multi-process `zeInit` abort (the multi-rank blocker) [HISTORICAL — fixed in 26.22]

Intel Compute Runtime **≥ 26.14** calls `abort()` during `zeInit`
whenever **≥2 processes** share the GPU (i.e. the normal MPI mode),
in `gmm_helper/resource_info.cpp:15`. CR **26.05** does not.

- **Deterministic**, reproducible with a pure-Level-Zero minimal program
  (no SYCL/Ginkgo/OGL): `gpu-diag/diag-mpi-l0` aborts at np≥2 on 26.18,
  passes on 26.05. Single process (np=1) works on 26.18.
- Persists through CR 26.18 + kernel 7.0.0-22.
- Upstream: [intel/compute-runtime#922](https://github.com/intel/compute-runtime/issues/922)
  (open, no fix). Full analysis: `findings/29`.
- No env-var workaround; an `MPI_Barrier` before `zeInit` does **not**
  help (the abort is in the driver, below OGL).

## Workaround: CR 26.05 LD-switch (no sudo, no system change)

Extract the CR 26.05 GPU backend and prepend it on `LD_LIBRARY_PATH`:
```bash
# one-time (no sudo):
apt download libze-intel-gpu1=26.05.37020.3-1 intel-opencl-icd=26.05.37020.3-1 intel-ocloc=26.05.37020.3-1
dpkg-deb -x <each>.deb ~/intel-cr-26.05/
# per shell:
source scripts/cr2605-shell.sh   # prepends ~/intel-cr-26.05/.../libze_intel_gpu, sets ONEAPI_DEVICE_SELECTOR=level_zero:0
```
Restores multi-rank L0/SYCL/Ginkgo/OGL. Verified post-reboot: clean 1→8
rank scaling. Details: `findings/27`, `scripts/cr2605-shell.sh`.

The system default remains CR 26.18 (deliberately not rolled back).
`sudo` is **not** passwordless here and the user is remote (no GUI polkit),
so all root changes are delivered as scripts the user runs themselves.

## GPU device-lost: behaviour & recovery

- **VRAM-OOM during a GPU run → `UR_RESULT_ERROR_DEVICE_LOST`** (e.g. ILU
  `ParIlu::generate_l_u` → `Csr::convert_to(Coo)` at the 32 GB ceiling).
- **It self-recovers** — a follow-up single-process `diag-l0` passes
  immediately, no reboot needed. The `intel_iommu=igfx_off` GRUB flag +
  fresh driver state clean up the DEVICE_LOST. (Contrast older notes about
  a stuck cascade — not observed on the post-reboot driver.)
- A **clean exception** (e.g. BJ `find_blocks` `gko::AllocationError`,
  16-EB request) does **not** device-lost — GPU stays healthy.
- Health gate between risky GPU runs: `gpu-diag/build/diag-l0` (exit 0 = OK).

## Quick reference
- `sycl-ls` shows both GPUs: `level_zero:0` = B70, `level_zero:1` = iGPU.
- CR version: `clinfo` / `sycl-ls` tail shows e.g. `26.18.38308.1`.
- Single-process GPU work: fine on 26.18. Multi-process: needs 26.05 switch.
