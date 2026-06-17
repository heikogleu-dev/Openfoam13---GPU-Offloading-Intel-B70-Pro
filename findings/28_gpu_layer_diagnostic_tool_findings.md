# Finding 28: Layer-by-Layer GPU Diagnostic — The CR 26.18 Race Is Not in Level Zero

## TL;DR — and an actionable workaround

A purpose-built three-layer diagnostic tool (L0 → SYCL → Ginkgo, with
both single-process and MPI multi-rank variants) ran against the GPU
that had just been declared "broken" by repeated OGL multi-rank
crashes ([Finding 27](27_cr2605_ld_switch_workaround.md)). Each layer
was tested in isolation. Results changed everything we thought we
knew about the CR 26.18 issue:

| Layer | Single-process | Multi-rank np=8 |
|---|---|---|
| **Pure Level Zero** (`zeInit`, `zeDriverGet`, `zeMemAllocDevice` 100 MiB) | ✅ | ✅ **all 8 ranks pass cleanly** |
| **Pure SYCL** with `ONEAPI_DEVICE_SELECTOR=level_zero:0` | ✅ | ✅ all 8 ranks pass cleanly |
| **Pure SYCL** without `ONEAPI_DEVICE_SELECTOR` | ✅ | ❌ `pthread_once` in `clGetPlatformIDs` |
| **Ginkgo** (`DpcppExecutor` + array alloc 200 MiB) without MPI barrier | ✅ | ❌ DEVICE_LOST in `raw_free` |
| **Ginkgo** with `MPI_Barrier` before `DpcppExecutor::create` | ✅ | ✅ **all 8 ranks pass cleanly** |

**The "CR 26.18 multi-rank race" is not in Level Zero. It is:**

1. **OpenCL platform enumeration races during SYCL init**, fixable by
   setting `ONEAPI_DEVICE_SELECTOR=level_zero:0` (which we already did
   for OpenFOAM — but OGL/MKL apparently triggers the OpenCL adapter
   load anyway through a different path).

2. **Unsynchronised rank arrival at `DpcppExecutor::create`**, fixable
   by inserting a single `MPI_Barrier(MPI_COMM_WORLD)` immediately
   before any SYCL/Ginkgo init. With the barrier, np=8 Ginkgo multi-rank
   runs 200 MiB allocations on every rank without a single failure.

This precisely re-frames the upstream debugging conversation: we are
no longer "blocked by an Intel CR multi-rank race we can't fix"
([Finding 25](25_cr_26.18_multirank_pthread_race.md) framing) but
"blocked by an OGL+SYCL init race that a one-line MPI_Barrier in OGL's
`ExecutorHandler` would close". That's an OGL-actionable fix.

## What "broken state" actually looked like

When the diagnostic was run shortly after the OGL multi-rank cascade
([Finding 27 section on broken-state cascade](27_cr2605_ld_switch_workaround.md#limitation-discovered--gpu-lockup-cascade-after-bj2-crash)),
the kernel log (`journalctl -k`) showed:

```
xe 0000:04:00.0: [drm] Tile0: GT0: Timedout job: seqno=4294967169, ...
                in Xwayland [4097]
xe 0000:04:00.0: [drm] Tile0: GT0: reset queued
xe 0000:04:00.0: [drm] Tile0: GT0: reset started
xe 0000:04:00.0: [drm] Tile0: GT0: reset done
xe 0000:04:00.0: [drm] Tile0: GT0: Timedout job: seqno=8193, ...
                in no process [-1]
xe 0000:04:00.0: [drm] Tile0: GT0: VM job timed out on non-killed execqueue
WARNING: drivers/gpu/drm/xe/xe_guc_submit.c:1596 at
         guc_exec_queue_timedout_job+0x493/0xc90 [xe]
```

The xe driver had already performed a Tile0 reset autonomously. The
"DEVICE_LOST" status reported by SYCL was the **driver's persistent
per-process knowledge of the prior context**, not actual hardware
lockup. A fresh process running our diagnostic 5-10 minutes later
saw the GPU fully functional: `sycl-ls` reported the B70, L0
allocated 100 MiB without issue, etc.

**Practical implication:** the "GPU broken state" cascade can be
shortened — instead of needing a reboot, just wait 5-10 minutes (xe
driver self-recovers via job timeout + GT reset) and start fresh
processes. The crashed processes' state is what's stuck, not the
hardware. The "non-killed execqueue" warning in dmesg suggests an
xe-driver cleanup bug worth filing upstream, but it self-clears.

## The diagnostic tool

Six binaries, each isolating one layer:

```
findings/code/gpu-diag/
├── CMakeLists.txt
├── diag-l0.cpp           Single-process pure Level Zero
├── diag-sycl.cpp         Single-process pure SYCL (with gpu_selector)
├── diag-ginkgo.cpp       Single-process Ginkgo (DpcppExecutor + BJ)
├── diag-mpi-l0.cpp       Multi-rank pure Level Zero
├── diag-mpi-sycl.cpp     Multi-rank pure SYCL
└── diag-mpi-ginkgo.cpp   Multi-rank Ginkgo
```

Each binary returns a distinct exit code per failing stage; logs are
fully readable and stage-tagged (`[ OK ] zeInit`, `[FAIL] zeMemAllocDevice 100MiB: 0x...`),
so a single run pinpoints where the stack fails.

Optional ENV-vars:

- `DIAG_BARRIER_BEFORE_INIT=1` — inserts `MPI_Barrier` immediately
  before zeInit / `sycl::device` / `DpcppExecutor::create`
- `DIAG_STAGGER_MS=50` — sleeps `rank * 50` ms before init (manual
  serialisation for race exploration)

The barrier flag is the one that resolved the multi-rank Ginkgo case.

## Reproduction — three commands that demonstrate the finding

After building the tools and running on a recovered GPU:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# 1. Pure L0 multi-rank: clean, no workaround needed
mpirun -np 8 ./diag-mpi-l0
# all 8 ranks: OK zeInit, OK driver/device, OK alloc 100 MiB, ALL DONE

# 2. Ginkgo multi-rank without barrier: fails at the 200 MiB allocation
mpirun -np 8 ./diag-mpi-ginkgo
# 3 ranks reach "OK 200 MiB", then DEVICE_LOST in raw_free for the rest

# 3. Same with MPI_Barrier before DpcppExecutor::create: clean
DIAG_BARRIER_BEFORE_INIT=1 mpirun -np 8 ./diag-mpi-ginkgo
# all 8 ranks: OK 1024, OK 200 MiB, ALL DONE
```

The same MPI launcher, the same Ginkgo build, the same CR 26.18 stack
— only the barrier separates pass from fail.

## What this means for the open issues

- [Finding 25 (CR 26.18 multi-rank pthread_once race)](25_cr_26.18_multirank_pthread_race.md):
  the race exists, but the layer it lives in is shallower than the L0
  loader. **Fix lives in OGL `ExecutorHandler`**, not in Intel CR.
- [Finding 27 (CR 26.05 LD-switch)](27_cr2605_ld_switch_workaround.md):
  still useful as a redundant safety net, but possibly unnecessary
  once OGL inserts the barrier.
- [Finding 02 (OGL distributed `find_blocks` underflow)](02_bj_blocksize_int_underflow.md):
  separate issue. Not a multi-rank init race.

## Upstream-actionable

Drafted PR text against hpsim/OGL — inserts a single
`MPI_Barrier(MPI_COMM_WORLD)` in `DeviceIdHandler::DeviceIdHandler`
(`include/OGL/DevicePersistent/ExecutorHandler.hpp`) before any SYCL
queue creation. To be tested + filed once GPU recovery permits one
more OpenFOAM multi-rank smoke run with the patch in place.

## Files

- `findings/code/gpu-diag/` — full source + CMakeLists for the six diagnostic binaries
- [`logs/gpu-diagnostic/`](../logs/gpu-diagnostic/):
  - `np8-WITHOUT-BARRIER-FAILS-at-raw_free.log` — the failing case
  - `np8-WITH-BARRIER-PASSES.log` — same code, +1 MPI_Barrier
  - `np4-passes.log` — np=4 baseline (passes without barrier)
