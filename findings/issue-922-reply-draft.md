# Draft reply for intel/compute-runtime issue #922 (ready to paste — NOT yet posted)

> Context: Intel (@ola2308, 2026-06-18) asked us to upgrade to CR 26.22.38646.4 +
> IGC 2.36.3 and report logs. Result below: **CR 26.22 RESOLVES the abort.**

---

Hi @ola2308 — thanks, good news: **CR 26.22.38646.4 fixes the multi-rank `resource_info.cpp:15` abort** for us.

I tested with our pure-Level-Zero reproducer (`diag-mpi-l0`, the one from the report — `zeInit → zeDriverGet → zeDeviceGet → zeContextCreate → zeMemAllocDevice`), via a non-invasive `LD_LIBRARY_PATH` switch (unpacked the release `.deb`, no system install):

| Stack (Level-Zero backend) | np=1 | np=8 |
|---|---|---|
| CR 26.18.38308.1 (system) | abort @ resource_info.cpp:15 | abort |
| CR 26.05.37020.3 (our pin) | OK | OK |
| **CR 26.22.38646.4 + IGC 2.36.3** | **OK** | **OK** |

`LD_DEBUG=libs` confirms the 26.22 backend is the one loaded:
```
calling init: ~/intel-cr-26.22/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
```
`sycl-ls` on the switched stack:
```
[level_zero:gpu] ... Intel(R) Arc(TM) Pro B70 Graphics 20.2.0 [1.15.38646+4]
```

I also re-ran our real workload (OpenFOAM 13 + OGL/Ginkgo, GKOCG + Multigrid,
`mpirun -np 8 foamRun -parallel -solver incompressibleFluid`, 7.1M-cell case) on
26.22 + IGC 2.36.3 — it runs correctly (converges, identical iteration counts) and
performance is unchanged vs the 26.05 pin (≈7.08 → 7.09 s/step). So 26.22 is a clean
fix with no regression for us.

**System info (as requested):**
```
$ lsb_release -a
Distributor ID: Ubuntu / Release: 26.04 (resolute)
$ uname -a
Linux ... 7.0.0-22-generic #22-Ubuntu SMP ... x86_64
GPU: Arc Pro B70 (BMG-G31), PCI 8086:e223
$ dpkg --list | grep -iE "igc|gmm|opencl|libze"   (system, pre-switch)
intel-igc-core-2        2.34.4
intel-igc-opencl-2      2.34.4
intel-opencl-icd        26.18.38308.1-0
libigdgmm12             22.10.0
libze-intel-gpu1        26.18.38308.1-0
libze1 / libze-dev      1.28.2-2
```
(The fix was verified by LD-switching the Level-Zero backend + IGC to the 26.22 /
2.36.3 release artifacts; the OpenCL ICD entry in `/etc/OpenCL/vendors` still points
at the system 26.18, but our workload is Level-Zero/SYCL, which uses the switched
26.22 backend as shown above.)

`dmesg` during the 26.22 multi-rank run: no GEM / xe / GPU-reset errors.

Happy to run any further confirmation. From our side this can be closed as fixed in
**26.22.38646.4** once you've reproduced — thank you for the quick turnaround.

---

## Notes for us (not for the issue)
- **Adoption path off the 26.05 pin:** either (a) keep the LD-switch but point it at
  `~/intel-cr-26.22` (proven here, `scripts/cr2622-shell.sh`), or (b) system-install
  CR 26.22 + IGC 2.36.3 via apt/dpkg (needs sudo; also fixes the OpenCL ICD path).
- **Perf:** IGC 2.36.3 / LLVM 17 gives **no speedup** on our SYCL kernels (7.1M
  identical) — our bottleneck is algorithmic (AMG scaling) + bandwidth + CPU-bound,
  not compiler codegen. See [knowledge/why-no-speedup-at-34M.md](../knowledge/why-no-speedup-at-34M.md).
