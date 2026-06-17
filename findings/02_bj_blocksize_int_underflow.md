# Finding 02: BJ maxBlockSize > 1 — size_t Underflow in `dpcpp::jacobi::find_blocks` (NOT OOM)

## Update — 2026-06-17: bug is in OGL distributed-matrix path, not Ginkgo `find_blocks` kernel

Re-tested under Ginkgo 2.0 with the [CR 26.05 LD-switch workaround](27_cr2605_ld_switch_workaround.md):

| Configuration | BJ(maxBlockSize=2) outcome |
|---|---|
| Ginkgo 2.0 standalone single-process, up to **36M rows** | ✅ runs clean (see [Finding 26](26_ginkgo_2.0_standalone_sweep.md)) |
| OGL+OpenFOAM multi-rank np=8, 34M cells (~4.25M per rank shard) | ❌ **identical** `0xFFFFFFFFFFFFFFFF` underflow |

Same Ginkgo 2.0 build, same `dpcpp::jacobi::find_blocks` kernel,
different result depending on whether the matrix arrives via plain
`gko::matrix::Csr` (standalone) or via OGL's distributed Schwarz
wrapper. **The bug surfaces specifically in the distributed
preconditioner generate path, not in the SYCL Jacobi kernel itself.**

The 1.10/1.11 stack triggered the same underflow because OGL always
runs distributed — we just never had a standalone reproducer to
isolate the kernel from the wrapper. The framing below is preserved
for historical context, but the upstream-actionable item for KIT/
`@nbeams` is now: investigate how OGL's distributed Schwarz/Matrix
path invokes `find_blocks` on per-rank shards.

---

## Important Correction (May 6, 2026)

Earlier versions of this document (filename `02_bj_maxblocksize_oom.md`)
attributed the BJ `maxBlockSize > 1` crash to a SYCL workspace allocation
bug (O(N × BS²)). **Direct VRAM measurement disproves that theory.**

## Direct Measurement

VRAM sampled at 5 Hz via `/sys/kernel/debug/dri/0/tile0/vram_mm` during runs
(see `profiling/vram_analysis.md` for methodology):

| Configuration | Peak VRAM | Available | Crash? |
|---|---|---|---|
| `BJ maxBlockSize=1` | **9.38 GB** | 27.9 GB | ✅ Runs (always 200-iter cap) |
| `BJ maxBlockSize=2` | **8.41 GB** | 27.9 GB | ❌ SIGABRT in generate |

Critical observation: **BJ(2) crashes with LESS VRAM than BJ(1)** — because
the crash happens during preconditioner *generate*, before the solver
allocates its Krylov workspace.

## Real Root Cause

From [`logs/vram-traces/log.vram-bj2`](../logs/vram-traces/log.vram-bj2)
line 285:

```
terminate called after throwing an instance of 'gko::AllocationError'
  what(): /opt/ogl-src/build/_deps/ginkgo-src/dpcpp/base/executor.dp.cpp:104:
          DPC++: failed to allocate memory block of 18446744073709551615B
```

`18446744073709551615` = 2⁶⁴ − 1 = **classic size_t underflow** (unsigned
representation of `-1`).

Stack trace (line 350):

```
gko::kernels::dpcpp::jacobi::find_blocks<double, int>
    → gko::array<bool>::array(executor, count = ~0ULL)
    → DpcppExecutor::raw_alloc(18446744073709551615)
```

`find_blocks` is computing a negative block count and casting to `size_t`
during the SYCL preconditioner generation phase. This is a deterministic
integer-arithmetic bug in Ginkgo 1.10's `dpcpp/preconditioner/jacobi/`
kernels for any `maxBlockSize > 1` on the 34M-cell mesh.

## Reproduction

- Hardware: Intel Arc Pro B70 Pro (BMG-G31)
- Stack: CR 26.05.37020.3-1, IGC 2.32.7, oneAPI 2026.0
- Mesh: 34M cells (~4.25M per rank, np=8)
- libOGL.so: rebuilt with `GINKGO_JACOBI_FULL_OPTIMIZATIONS=ON`
  + `-fp-model=precise` (does NOT fix this — separate code path, see below)

`fvSolution` p-block:

```
solver       GKOCG;
executor     sycl;
preconditioner { preconditioner BJ; maxBlockSize 2; }   // 4, 8, 16 — same crash
```

Crash deterministic at the `Generate preconditioner BJ<double> MaxBlockSize 2`
log line, before any solve iteration.

## Why `GINKGO_JACOBI_FULL_OPTIMIZATIONS=ON` Does Not Help

This flag affects the CUDA Jacobi adaptive-precision optimisation path,
not the SYCL block-counting kernel where the underflow originates. We
verified by rebuilding with the flag set and re-running — identical
crash, identical underflow value.

## Status

- Raw logs and VRAM traces preserved in
  [`logs/vram-traces/`](../logs/vram-traces/)
- Reproduction is 100% deterministic on this hardware/mesh size
- Upstream issue body ready to file at
  `ginkgo-project/ginkgo` (planned — see `findings/11_ginkgo_issue_body.md`
  for the existing draft, to be extended with this new evidence)

## What This Reveals About Earlier Hypotheses

| Hypothesis | Status |
|---|---|
| "SYCL workspace allocation O(N × BS²)" | ❌ Refuted by VRAM measurement |
| "Real VRAM exhaustion at 32 GB limit" | ❌ Refuted (peak 8.4 GB / 27.9 GB) |
| **"size_t underflow in `find_blocks` block-counting"** | ✅ Confirmed by raw allocation size |

The earlier hypotheses were architecturally plausible but wrong. Direct
measurement is the only way to know.
