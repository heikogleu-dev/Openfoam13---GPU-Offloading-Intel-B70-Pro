# Test Plan — After GPU Recovery (Reboot or iGPU Kill)

The CR 26.05 LD-switch ([Finding 27](../findings/27_cr2605_ld_switch_workaround.md))
opens up the full multi-rank OGL pipeline again, but the BJ(2) crash
leaves the GPU in a stuck `DEVICE_LOST` cascade that requires reboot
or iGPU kill to recover. Once recovered, walk through these tests
**one preconditioner per session** to avoid burning the GPU on a single
crash.

## Recommended order (cheapest first)

Each test is one `endTime=1` smoke run, ~2 min on the GPU. Save the
log under `log.post-recovery/test-NAME.log` and check for either
"Solving for p," (success) or the failure signature, then **reboot**
between tests until the cascade root cause is understood.

### 1. Sanity — BJ(1) baseline (should work, 94 s/step)

```bash
source scripts/cr2605-shell.sh
sed -i 's/preconditioner  .*/preconditioner  BJ;/' system/fvSolution
grep -q "maxBlockSize" system/fvSolution || sed -i '/preconditioner  BJ;/a\            maxBlockSize    1;' system/fvSolution
sed -i 's/maxBlockSize    [0-9]*;/maxBlockSize    1;/' system/fvSolution
# clean processor* time dirs, then:
mpirun -np 8 foamRun -parallel -solver incompressibleFluid > log.post-recovery/test-BJ1.log 2>&1
```

Expected: ✅ ExecutionTime ~94 s, 201 iter (reproduces Finding 27 baseline).

### 2. BJ(maxBlockSize=4) — sweet spot block size, should converge faster

```bash
sed -i 's/maxBlockSize    1;/maxBlockSize    4;/' system/fvSolution
# clean + run
```

Expected outcomes:
- ✅ runs → BJ(2)-class bug is `maxBlockSize=2`-specific or doesn't trigger here; record iter count
- ❌ same `0xFFFFFFFFFFFFFFFF` underflow → bug is in the distributed wrapper for any `maxBlockSize > 1`

### 3. BJ(8), BJ(16) — if BJ(4) works

Each one separate session. We expect them all to either work or all fail
the same way as BJ(4); the gradient of failure is the upstream-relevant data.

### 4. ILU — the strong preconditioner that Finding 24 reached but VRAM-OOM'd

Set up:
```bash
sed -i 's/preconditioner  BJ;/preconditioner  ILU;/' system/fvSolution
sed -i '/maxBlockSize/d' system/fvSolution
```

Expected: VRAM peak observed live via Ubuntu Resources tool. Reference
from Finding 24 (CR 26.05 era): ~26.5 GB plateau then >32 GB spike at
`Csr::convert_to(Coo)`. The standalone bytes/row data (Finding 26)
suggests ILU fits within 27 GB at 34M cells (~9 GB standalone, ~30 GB
with OGL distributed overhead — right at the limit).

### 5. Multigrid — only if a smaller mesh is available

Standalone bytes/row (Finding 26) is ~1027 bytes/row → at 34M cells
~35 GB **above** the 32 GB ceiling. Likely OOM on the full mesh.
Worth trying once with VRAM monitoring to confirm and characterize.

### 6. ICT (`int32` overflow expected) — if Ginkgo is rebuilt with workaround

Requires Ginkgo rebuild with `-fno-sycl-id-queries-fit-in-int`. Not
done yet — separate work item.

## Between tests: GPU recovery checklist

If the test crashes (`DEVICE_LOST`, `pthread_once`, any `terminate`):

1. Wait 60 s — sometimes the xe driver self-recovers
2. Sanity-test with BJ(1): `mpirun -np 8 foamRun ...` — if it returns
   `DEVICE_LOST` too, GPU is stuck
3. iGPU/xe kill (Heiko's rule: max 2 per session) **or** reboot
4. After recovery: re-decompose if cold, otherwise continue

## VRAM monitoring during ILU/Multigrid tests

VRAM stats via `xe debugfs` need a fresh `chmod` after reboot. From a
shell with X access (or via SSH with display forwarding for the
polkit prompt):

```bash
pkexec sh -c 'chmod a+x /sys/kernel/debug /sys/kernel/debug/dri \
    /sys/kernel/debug/dri/0 /sys/kernel/debug/dri/0/tile0 && \
    chmod a+r /sys/kernel/debug/dri/0/tile0/vram_mm'
```

Then the existing sampler (`scripts/vram-sampler.sh` if added, or the
inline loop from Finding 22) can run as user.

Alternative: Ubuntu Resources tool live-reading during the run, paired
with the standalone allocation counts from `vram-sweep-v2-with-free-tracking.csv`
for cross-check.

## Pioneer-output goals from this round

1. **Confirm BJ(2-16) Multi-Rank behaviour** — either all fail
   (distributed-wrapper bug for any BS>1) or BJ(2) is special. Either
   way → useful narrowing for upstream.
2. **ILU performance number on 34M cells if it fits** — would be
   the first practical strong-preconditioner GPU CFD step on BMG-G31.
3. **Comparison to CPU GAMG (35.7 s/step)** — if ILU converges in
   say 20 iter vs BJ(1) 200 iter, the total step time could finally
   beat CPU. That's the GPU-WIN scenario we've been after.

## Files

- [`scripts/cr2605-shell.sh`](cr2605-shell.sh) — activation
- [`logs/cr26.05-switch-test/`](../logs/cr26.05-switch-test/) — last session's results
- [`findings/27`](../findings/27_cr2605_ld_switch_workaround.md) — context
