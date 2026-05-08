# Finding 22: GMRES VRAM Pressure on BMG-G31 — Hardware Limit, Not Ginkgo Bug

## TL;DR

GKOGMRES on a 34M-cell mesh + np=8 hits the **hardware VRAM limit** on
B70 Pro (27.87 GB / 27.9 GB usable), triggering
`UR_RESULT_ERROR_DEVICE_LOST`.

This is **not a Ginkgo bug** — it is a real OOM. But it is an important
hardware-pioneer datapoint: 32 GB VRAM is not sufficient for GMRES on
automotive-scale CFD meshes with the current Ginkgo distributed-matrix
allocation pattern.

## Initial (Wrong) Hypothesis

After Stufe 4d, the working hypothesis was: "GKOGMRES is structurally
broken on BMG-G31" — every GKOGMRES run crashed with `DEVICE_LOST`,
including with the simplest `BJ(maxBlockSize=1)` preconditioner. This
was wrong.

## Test Method — Dual VRAM Measurement

Two independent VRAM sources sampled in parallel during the same run:

| Source | Reads | Limitation |
|---|---|---|
| `xe debugfs` `/sys/kernel/debug/dri/0/tile0/vram_mm` field `usage:` | dedicated VRAM only | hardware truth, ignores USM spillover |
| Ubuntu Resources tool — GPU 1 (Battlemage), VRAM Used | logical incl. USM | counts spillover into host RAM |

Plus `/proc/meminfo` (system RAM) and `ps -o rss,comm` (foamRun cumulative
RSS) for cross-correlation.

## Results

| Configuration | xe debugfs (dedicated) | Resources tool | System RAM | foamRun RSS | Outcome |
|---|---|---|---|---|---|
| `GKOGMRES + BJ(1) krylovDim=30` | **27.87 GB** | 35.1 GB | +49 GB | 53.3 GB | DEVICE_LOST |
| `GKOGMRES + BJ(1) krylovDim=5`  | **27.86 GB** | 35.07 GB | +49 GB | 53.4 GB | DEVICE_LOST |

Hardware VRAM available: 27.9 GB (32 GB − driver/firmware reservation).

## Pioneer Discoveries

### 1. USM Spillover into DDR5 confirmed and quantified

`Resources tool − xe debugfs` = `35.1 − 27.87` ≈ **7.2 GB transparently
spilled into Host RAM** via the SYCL Unified Shared Memory mechanism.

This confirms that since Compute Runtime 26.01, when dedicated VRAM is
full, USM allocations go transparently into the host RAM. The xe
debugfs `vram_mm` counter shows only what is physically in GDDR6, while
the Ubuntu Resources tool shows the SYCL-logical VRAM (incl. spillover).
**Both readings are correct for what they report — the discrepancy is
the spillover.**

### 2. krylovDim is NOT the dominant memory factor

Reducing `krylovDim` from 30 → 5 (6× smaller Krylov basis) yields:

- Expected savings (Krylov vector basis only): ~6.6 GB
  (5 × `nCells_per_rank × nVecs × 8 bytes`)
- Actual savings: 0.01 GB

GMRES allocates ~26 GB of *fixed overhead* upfront, **independent of
`krylovDim`**. Likely culprits:

- Distributed-matrix duplicates (with `ranksPerGPU=8`, possibly one
  matrix copy per rank against the same GPU)
- Ginkgo-internal workspaces sized by total non-zero count, not iteration
  count
- Hessenberg matrix and pre-allocated maximum history buffers

This means **`krylovDim` is not a usable workaround** for the OOM.

### 3. DEVICE_LOST is poor diagnostics

When VRAM hits 27.87 GB, the Ginkgo allocation request fails. The SYCL
backend reports `UR_RESULT_ERROR_DEVICE_LOST` rather than a clean
`gko::AllocationError`. This is a **diagnostics quality issue**, not a
bug — but suboptimal user experience for hitting the OOM ceiling.
Constructive feedback for upstream Ginkgo: catching the L0 allocation
failure path explicitly and re-raising it as `gko::AllocationError`
would help users distinguish OOM from real driver issues.

## Implications

### For Battlemage CFD Pioneers

- **32 GB VRAM is insufficient** for GMRES on automotive-scale CFD
  (34M cells, np=8, distributed)
- CG-based solvers stay within bounds (~9 GB peak measured for
  CG + BJ(1), see [Finding 02 / vram_analysis](../profiling/vram_analysis.md))
- USM spillover masks OOM transparently until SYCL fails internally;
  the system shows ~50 GB foamRun RSS which is the real working set

### For Ginkgo Project (constructive feedback)

- Better OOM diagnostics: distinguish allocation failure from genuine
  device errors at the SYCL boundary
- Document the per-rank distributed-matrix memory footprint so users
  can size hardware accordingly

### For petsc4Foam Path (Plan B)

- Hypre BoomerAMG has a similar memory profile (large coarsening
  hierarchy)
- petsc4Foam tests on the full MR2 case should monitor VRAM with this
  same dual-measurement methodology
- Plan B variant: use a mesh-subset (e.g. 8M cells) for first
  petsc4Foam pioneer test on this hardware

## Reproduction Methodology

```bash
# One-time: open debugfs read access (resets at reboot)
pkexec sh -c 'chmod a+x /sys/kernel/debug /sys/kernel/debug/dri \
   /sys/kernel/debug/dri/0 /sys/kernel/debug/dri/0/tile0 \
&& chmod a+r /sys/kernel/debug/dri/0/tile0/vram_mm'

# Sampler — runs as user, no further pkexec
(while true; do
    TS=$(date +%s.%N)
    VRAM=$(grep "^  usage:" /sys/kernel/debug/dri/0/tile0/vram_mm \
             | awk '{print $2}')
    TOTAL=$(grep "^MemTotal" /proc/meminfo | awk '{print $2}')
    AVAIL=$(grep "^MemAvailable" /proc/meminfo | awk '{print $2}')
    RAM=$(( (TOTAL - AVAIL) * 1024 ))
    RSS=$(ps -eo rss,comm | awk '$2 == "foamRun" {sum += $1} END {print sum*1024}')
    echo "$TS,$VRAM,$RAM,$RSS"
    sleep 0.2
done) > vram-trace.csv &

# Run case, parallel:
# - Read CSV column 2 (xe debugfs) for dedicated VRAM
# - Read Ubuntu Resources tool GPU 1 VRAM for logical (incl. USM)
# - Discrepancy = USM spillover
```

## Files

- [`logs/stufe4-ginkgo111/memory-trace-1.11.csv`](../logs/stufe4-ginkgo111/memory-trace-1.11.csv)
  — kdim=30 full trace (384 samples, 86 s)
- [`logs/stufe4-ginkgo111/memory-trace-kdim5.csv`](../logs/stufe4-ginkgo111/memory-trace-kdim5.csv)
  — kdim=5 verification (343 samples, 77 s)
- [`logs/stufe4-ginkgo111/teil-a-1.11.log`](../logs/stufe4-ginkgo111/teil-a-1.11.log)
  — DEVICE_LOST event log
- [`logs/stufe4-ginkgo111/teil-b-kdim5-1.11.log`](../logs/stufe4-ginkgo111/teil-b-kdim5-1.11.log)
  — DEVICE_LOST persists with smaller krylovDim
