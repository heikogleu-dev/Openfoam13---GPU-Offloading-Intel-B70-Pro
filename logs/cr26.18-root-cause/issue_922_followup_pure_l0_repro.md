# Follow-up for intel/compute-runtime#922 — persists through CR 26.18, kernel-independent, pure-Level-Zero minimal reproducer

Update on GSD-12696. Three new data points that should make this much
easier to bisect on Intel's side.

## 1. Still present in CR 26.18 (latest)

The `resource_info.cpp:15` abort is unchanged from 26.14 through
**26.18.38308.1** (current latest, June 2026). Three releases, no fix.
The 26.18 driver prints the abort with an empty `__FILE__`:

```
Abort was called at 15 line in file:
```

but the stack and behaviour are identical to the 26.14
`gmm_helper/resource_info.cpp:15` abort reported originally.

## 2. A newer kernel does NOT fix it

The original report was on kernel `7.0.0-15-generic`. We are now on
**`7.0.0-22-generic`** (xe module `srcversion ACC2C75180429CFD259EDF2`)
and the abort is identical. This weakens the "ABI mismatch with the xe
kernel module" hypothesis from the original report — a newer xe module
did not change anything.

GMM userspace: `libigdgmm12 22.10.0`.

## 3. Pure-Level-Zero minimal reproducer (no SYCL / MPI-runtime-SYCL needed)

The original reproducer was a full SYCL + MPI application. Here is a
~120-line program that reproduces the abort with **only `libze_loader`
+ MPI** — no SYCL, no oneMKL, no application framework. Each rank just
does `zeInit → zeDriverGet → zeDeviceGet → zeContextCreate →
zeMemAllocDevice`:

```bash
mpirun -np 2 ./diag-mpi-l0     # ABORT at resource_info.cpp:15
mpirun -np 1 ./diag-mpi-l0     # OK
```

Full source: https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/code/gpu-diag/diag-mpi-l0.cpp

Characterisation matrix (B70 Pro, BMG-G31, CR 26.18, kernel 7.0.0-22):

| Config | Result |
|---|---|
| np=1 | OK |
| np=2 | abort |
| np=8 | abort |
| np=8, ranks staggered 200 ms apart | abort (timing does not help) |
| np=8, `LD_LIBRARY_PATH` → CR 26.05 `libze_intel_gpu.so` | all ranks OK |

Staggering not helping indicates this is **deterministic**, not a
start-up timing race: the second process to touch the device aborts in
GMM resource-info init regardless of when it arrives.

## 4. Env-var workarounds that do NOT work

Tried the IPC/GMM-related NEO debug keys as environment variables
(plain names) with `diag-mpi-l0 -np 8` on 26.18 — all still abort:

- `EnablePidFdOrSocketsForIpc=1` / `=0`
- `ForceIpcSocketFallback=0`
- `EnableIpcSocketFallback=0`

(Presumably these are not honoured as plain env vars in a release
build.) If there is a supported runtime switch to make GMM
resource-info init multi-process-safe, that would be a great interim
workaround — otherwise the only fix remains pinning `libze-intel-gpu1`
to 26.05.

## Why this matters

Multi-process single-GPU is the **normal** execution mode for MPI HPC
(CFD, etc.), not an edge case. Any Level-Zero MPI workload on
Battlemage is currently blocked on CR ≥ 26.14 unless the user knows to
downgrade. The pure-L0 reproducer above should let you confirm and
bisect without any of our CFD stack.

Happy to run debug builds or `NEO_*`-prefixed debug keys if you can
point at the right enable mechanism.
