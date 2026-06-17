## Follow-up on Issue #2015 — `find_blocks` underflow is in the OGL distributed-matrix path, not the SYCL Jacobi kernel

Picking up from our previous comments and your [PR #2023](https://github.com/ginkgo-project/ginkgo/pull/2023) merge.

We built OGL against Ginkgo 2.0 `develop` (via the in-progress
[hpsim/OGL#168](https://github.com/hpsim/OGL/pull/168) + two minimal
patches — separately submitted to OGL upstream) and re-ran the
`BJ(maxBlockSize=2)` reproducer on BMG-G31, with full results in our
public repository:

https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro

**Headline:** the `find_blocks` `size_t` underflow that we originally
reported for Ginkgo 1.10 / 1.11 still triggers under Ginkgo 2.0, but
**only** through OGL's distributed-matrix wrapper. A standalone
single-process Ginkgo 2.0 test program that exercises the same
`dpcpp::jacobi::find_blocks` kernel directly — up to 36 million rows
on a 2D Poisson 5-point matrix — runs cleanly (see
[finding 26](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/26_ginkgo_2.0_standalone_sweep.md)).

| Configuration | Ginkgo backend | Path | BJ(2) outcome |
|---|---|---|---|
| Standalone test program, single process | dpcpp 2.0.0 | plain `gko::matrix::Csr` | ✅ runs at 1M, 4M, 16M, 36M rows |
| OGL + OpenFOAM, np=8, 34M cells | dpcpp 2.0.0 (same install) | distributed Schwarz wrapper, ~4.25M-row per-rank shard | ❌ identical `0xFFFFFFFFFFFFFFFF` underflow |

This precisely localises the bug. The SYCL Jacobi kernel itself was
fixed somewhere between Ginkgo 1.11 and 2.0 (we verified BJ(2), BJ(4),
BJ(8), BJ(16) all work standalone in 2.0). What still fails is the
path where OGL feeds a per-rank `Csr` shard into the distributed
preconditioner generate via `experimental::distributed::preconditioner::Schwarz`.

A possible hypothesis worth investigating: per-rank shards in our case
are ~4.25M rows — within the range where standalone runs cleanly —
but the shard may have characteristics (sparsity pattern, missing
diagonal coverage, empty row blocks) that the standalone Poisson
matrix doesn't reproduce, triggering the block-counting underflow only
in that case.

Happy to:

1. Run additional configurations with debug builds if helpful for
   narrowing the kernel call site
2. Dump the actual per-rank `Csr` shard that OGL feeds in (matrix
   market format) so the standalone reproducer can be made
   bit-identical to what OGL sees

— Heiko

P.S. The OGL multi-rank path is also currently gated by an unrelated
Compute Runtime 26.18 `pthread_once`/`zeInit` race on Intel BMG-G31;
we work around it via a user-side `LD_LIBRARY_PATH` switch to the
`libze_intel_gpu.so` shipped with CR 26.05 — see [finding 27](https://github.com/heikogleu-dev/Openfoam13---GPU-Offloading-Intel-B70-Pro/blob/main/findings/27_cr2605_ld_switch_workaround.md)
for the reproducer and workaround. Not a Ginkgo issue.
