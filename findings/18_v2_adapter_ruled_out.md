# Finding 18: Level Zero V2 Adapter Ruled Out as Cause of `find_blocks` Underflow

## Hypothesis Tested

The Battlemage-default V2 Level Zero adapter (since CR 26.01) supports only
immediate in-order command lists. **Hypothesis:** the V2 immediate-mode
constraint might cause the integer underflow in `dpcpp::jacobi::find_blocks`
for `BJ(maxBlockSize>1)` by leaving an intermediate value negative before the
`size_t` cast.

## Test Setup

- Hardware: Intel Arc Pro B70 Pro (BMG-G31, Device `0xE223`)
- Stack: CR 26.05.37020.3-1, IGC 2.32.7, oneAPI 2026.0
- libOGL.so: existing build with `-fp-model=precise` +
  `GINKGO_JACOBI_FULL_OPTIMIZATIONS=ON`
- Mesh: 34M cells (~4.25M per rank, np=8)
- `fvSolution`: `solver GKOCG; executor sycl; preconditioner BJ; maxBlockSize 2;`
- Environment:

```
export SYCL_UR_USE_LEVEL_ZERO_V2=0    # force V1 adapter
export SYCL_UR_TRACE=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

## Adapter Selection Verification

Both adapter libraries available in `/opt/intel/oneapi/`:

- `libur_adapter_level_zero.so` (V1)
- `libur_adapter_level_zero_v2.so` (V2)

`SYCL_UR_TRACE=1` confirms platform `oneAPI Unified Runtime over Level-Zero`
across all 8 ranks with device `[0xe223]`. A direct V1/V2 string is not
emitted in the trace — the env var toggle is the documented behaviour and
the only available switch.

## Result: Identical Failure

Both V1 and V2 adapters produce:

- Same crash position: `dpcpp/base/executor.dp.cpp:104`
- Same allocation request: `18446744073709551615 B` = `2^64 − 1`
- Same crash phase: preconditioner generate, before first solve iteration
- VRAM peak within sample noise (~60 MB difference)

| Configuration | VRAM Peak | Δ over baseline | Crash address | Outcome |
|---|---|---|---|---|
| BJ(2) with V2 (default) | 8.41 GB | 7.22 GB | `executor.dp.cpp:104` | SIGABRT, `0xFFFFFFFFFFFFFFFF` |
| BJ(2) with V1 (`V2=0`) | 8.47 GB | 7.26 GB | `executor.dp.cpp:104` | SIGABRT, `0xFFFFFFFFFFFFFFFF` |

## Conclusion

The `size_t` underflow in `dpcpp::jacobi::find_blocks` is **not** caused by
the Level Zero V2 adapter. The bug sits inside Ginkgo's SYCL preconditioner
generate path itself, independent of which Unified Runtime adapter selects
command queues.

## Implications for Upstream Debugging

This narrows the search space for the `find_blocks` underflow report:

- **Rules out:** L0 Runtime adapter layer (V1 vs V2)
- **Still in scope:** `dpcpp/preconditioner/jacobi/` integer arithmetic
- **Still in scope:** SYCL queue submission semantics inside Ginkgo
- **Still in scope:** kernel block-counting math for non-trivial `maxBlockSize`

## Files

- [`logs/v1-adapter-test/log.bj2-v1adapter`](../logs/v1-adapter-test/log.bj2-v1adapter)
  — full mpirun output with V1 adapter
- [`logs/v1-adapter-test/vram-bj2-v1adapter.csv`](../logs/v1-adapter-test/vram-bj2-v1adapter.csv)
  — VRAM trace, 84 s, 405 samples

## Status

Datapoint contributed to `ginkgo-project/ginkgo#2015` follow-up.
