# Finding 27: CR 26.05 LD_LIBRARY_PATH Workaround — Multi-Rank OGL Restored Without System Rollback

## TL;DR

[Finding 25](25_cr_26.18_multirank_pthread_race.md) documented that CR
26.18 multi-rank breaks via a `pthread_once`/`zeInit` race on BMG-G31.
Direct rollback to CR 26.05 was rejected (user preference: stay on the
Intel-shipped current version).

**This finding documents a user-space workaround that runs CR 26.05's
Intel-GPU Level-Zero backend (`libze_intel_gpu.so`) under the system's
CR 26.18 loader — switched per-run via `LD_LIBRARY_PATH`.** No `sudo`,
no system modification, fully reversible. Multi-rank OGL on BMG-G31
works again.

## Setup (one-time, user-side only)

```bash
# Download .deb packages from Ubuntu universe repo (no sudo needed)
cd /tmp/cr26.05-download
apt download \
    intel-opencl-icd=26.05.37020.3-1 \
    libze-intel-gpu1=26.05.37020.3-1 \
    intel-ocloc=26.05.37020.3-1

# Extract to user-owned directory (no sudo needed)
mkdir -p /home/heiko/intel-cr-26.05
for d in *26.05*.deb; do
    dpkg-deb -x "$d" /home/heiko/intel-cr-26.05/
done
# Total size: ~83 MB
```

That gives:
```
/home/heiko/intel-cr-26.05/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020
/home/heiko/intel-cr-26.05/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
/home/heiko/intel-cr-26.05/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
/home/heiko/intel-cr-26.05/usr/lib/x86_64-linux-gnu/libocloc.so
```

## Usage (per-run, env var only)

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export LD_LIBRARY_PATH=/home/heiko/intel-cr-26.05/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
mpirun -np 8 foamRun -parallel -solver incompressibleFluid
```

The system loader resolves `libze_intel_gpu.so.1` from the prepended
path before searching `/usr/lib/...`, so CR 26.05's Intel GPU backend
takes precedence while the rest of the L0 loader (`libze_loader.so.1`,
from `libze1` package) stays on the system version.

## Verification — multi-rank BJ(1) works again

```
[OGL LOG][lduLduBase.hpp:106] Initialising OGL
	Ginkgo version: 2.0.0 ( develop)
BJsyclGKOCG:  Solving for p, Initial residual = 1, Final residual = 0.8738505, No Iterations 201
BJsyclGKOCG:  Solving for p, Initial residual = 0.01545088, Final residual = 0.00471455, No Iterations 202
BJsyclGKOCG:  Solving for p, Initial residual = 0.00170075, Final residual = 0.0007565485, No Iterations 201
ExecutionTime = 93.84717 s  ClockTime = 94 s
```

**RC=0. 201 iterations, ExecutionTime 93.8 s for 1 timestep on
34M cells / np=8.** Identical iteration count and ~same performance
as the pre-CR-26.18 era (Finding 02 baseline was 53 s/step for the
mean of steady-state steps 8-10; 94 s for the init-dominated first
step is consistent).

## Pioneer-relevant: BJ(maxBlockSize=2) crash IS in OGL distributed-matrix path

The follow-up Multi-Rank test of `BJ(maxBlockSize=2)` immediately
reproduced the **identical Finding 02 underflow** that Ginkgo 2.0
standalone runs do NOT have:

```
terminate called after throwing an instance of 'gko::AllocationError'
  what(): dpcpp/base/executor.dp.cpp:104:
          DPC++: failed to allocate memory block of 18446744073709551615B
```

But:

| Configuration | BJ(2) outcome |
|---|---|
| Ginkgo 2.0 **standalone**, single process, up to 36M rows | ✅ runs clean (see [Finding 26](26_ginkgo_2.0_standalone_sweep.md)) |
| OGL+OpenFOAM **multi-rank np=8**, 34M cells (~4.25M per rank shard) | ❌ identical `0xFFFFFFFFFFFFFFFF` underflow |

**Same Ginkgo 2.0 build, same `find_blocks` kernel. Different result.**
The bug is no longer in `dpcpp::jacobi::find_blocks` itself — it's
triggered by the distributed-matrix path that OGL feeds in: either
the per-rank `Csr` shard (~4.25M rows, smaller than the standalone
36M test that succeeded) is somehow pathological, or the
distributed-matrix wrapper invokes `find_blocks` in a different way.

This is a precise upstream-actionable finding for the next pass at
[ginkgo-project/ginkgo#2015](https://github.com/ginkgo-project/ginkgo/issues/2015):
the bug is not in the SYCL Jacobi kernel — it's in how the distributed
Schwarz/Matrix path feeds it.

## Limitation discovered — GPU lockup cascade after BJ(2) crash

After the BJ(2) crash, **all subsequent OGL runs on the same GPU
state surface as `UR_RESULT_ERROR_DEVICE_LOST`**, including:

- BJ(1) sanity re-test (had run cleanly 5 minutes earlier)
- BJ(4), BJ(8), BJ(16) (untested algorithm paths)
- ILU, ICT, Multigrid
- Even single-process standalone Ginkgo tests

The GPU enters a stuck state that persists until either:
- iGPU/xe driver kill (user constraint: max 2/session)
- System reboot

Practical consequence: each failed OGL multi-rank test currently
"burns" the GPU for the rest of the session unless recovered.
Future test sweeps should run **one preconditioner per session**
or include explicit between-test GPU recovery, until the underlying
cascade is understood.

## What this changes (status update across the repo)

- [Finding 25](25_cr_26.18_multirank_pthread_race.md): CR 26.18 race
  still real but now has a user-side workaround
- [Finding 02](02_bj_blocksize_int_underflow.md): the `find_blocks`
  underflow narrative needs the new "OGL distributed path, not Ginkgo
  kernel" framing
- [Finding 26](26_ginkgo_2.0_standalone_sweep.md): standalone
  algorithm-level results stand; the gap between "standalone OK" and
  "OGL multi-rank crash" is now precisely the OGL distributed wrapper
- Multi-rank tests for BJ(4-16), ILU, ICT, Multigrid in the actual
  OpenFOAM pipeline are now **technically possible** but await
  GPU recovery + isolated per-session runs

## Files

- [`logs/cr26.05-switch-test/`](../logs/cr26.05-switch-test/)
  contains all multi-rank test logs from this finding
- LD-switch utility script: [`scripts/cr2605-shell.sh`](../scripts/cr2605-shell.sh)
  for convenient per-session activation
