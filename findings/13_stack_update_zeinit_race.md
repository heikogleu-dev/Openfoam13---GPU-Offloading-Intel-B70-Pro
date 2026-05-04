# Bug: Compute Runtime 26.14 incompatible with multi-rank OGL/OpenFOAM

## TL;DR

Updating Intel Compute Runtime from 26.05 (Ubuntu Universe) to 26.14 (Intel
release) **breaks multi-rank OpenFOAM+OGL runs** with a `resource_info.cpp:15`
abort. Single-process SYCL works fine. **Rolled back to 26.05** — the only
working configuration.

IGC v2.32.7 alone is safe and brings measurable FP32 improvement.

## Update Attempted

- Compute Runtime: 26.05.37020.3-1 → **26.14.37833.4** (Intel release .deb)
- IGC: not previously installed → **v2.32.7+21184** (Intel release .deb)
- libigdgmm12: 22.9.0+ds1-1 (Ubuntu) → 22.9.0 (Intel)

## Symptom Sequence

1. Single-process tests passed:
   - `sycl-ls`: B70 Pro detected normally
   - FP64 bench: 1346 GFLOPS (was 1335) — fine
   - FP32 bench: **14464 GFLOPS** (was 12377) — **+17% improvement**

2. Multi-rank `mpirun -np 8 foamRun -parallel`:
   - Pre-reboot: All 8 ranks abort in `zeInit` (race condition)
   - Post-reboot: Init passes, U-solver works, **crash on first p-solve**:

   ```
   Abort was called at 15 line in file:
   ../../neo/shared/source/gmm_helper/resource_info.cpp
   ```

3. Tested both newly-rebuilt libOGL.so and the pre-update backup
   libOGL.so — same crash either way. Confirms the bug is in NEO/libze
   26.14, not in OGL/Ginkgo.

## Root Cause Analysis

The abort at `resource_info.cpp:15` is in NEO's `Gmm::Gmm()` constructor
assertion. Triggered during GPU resource allocation when 8 MPI ranks
allocate device memory concurrently. The newer NEO 26.14 expects something
in the host environment (kernel API? gmm_helper version?) that does not
match Ubuntu 26.04's `xe` kernel module.

Tested mitigations that did NOT help:
- Reboot (only partially fixed the race; deeper bug remained)
- Intel's `libigdgmm12 22.9.0` instead of Ubuntu's `+ds1-1` variant
- Pre-update OGL build vs newly rebuilt OGL

## Workaround / Resolution

**Roll back Compute Runtime + libze to 26.05** while keeping IGC v2.32.7:

```bash
pkexec apt install -y --reinstall --allow-downgrades --allow-change-held-packages \
    intel-opencl-icd=26.05.37020.3-1 \
    libze-intel-gpu1=26.05.37020.3-1 \
    intel-ocloc=26.05.37020.3-1 \
    libigdgmm12=22.9.0+ds1-1
pkexec apt-mark hold intel-opencl-icd libze-intel-gpu1 intel-ocloc
```

The `apt-mark hold` is critical — Ubuntu PackageKit/unattended-upgrades
silently re-applied `libigdgmm12` between install and reboot, complicating
the diagnosis.

## Post-Rollback State (Verified Working)

- intel-opencl-icd: 26.05.37020.3-1 (Ubuntu)
- libze-intel-gpu1: 26.05.37020.3-1 (Ubuntu)
- intel-ocloc: 26.05.37020.3-1 (Ubuntu)
- libigdgmm12: 22.9.0+ds1-1 (Ubuntu)
- **intel-igc-core/opencl: 2.32.7** (Intel — kept, no compatibility issues)

BJ maxBlockSize=1 sweep confirmed working at **53.8 s/step** (matches
pre-update 53.5 s/step baseline within noise).

## What This Costs

The +17% FP32 improvement seen in single-process bench was a false signal —
multi-rank workload crashed before benefiting. The rollback keeps the safer
26.05 stack at the original 12377 GFLOPS FP32. No production loss.

## Lessons

1. **Single-process bench after a driver update is misleading** — always
   verify with the actual multi-rank workload before declaring success
2. **Plan a reboot after Intel `.deb` updates** to ensure userland and
   kernel module are in sync
3. **`apt-mark hold` after Intel `.deb` install** prevents PackageKit from
   silently re-applying Ubuntu-stock packages
4. **Bisect by rolling back individual components** — the resource_info
   crash was misattributed to libigdgmm12 at first; only after testing
   Intel's variant did the actual cause (libze 26.14) become clear

## Filed Upstream

Will file a NEO/compute-runtime issue with full reproduction:
- https://github.com/intel/compute-runtime/issues
