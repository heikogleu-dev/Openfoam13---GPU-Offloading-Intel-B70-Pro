# Intel B70 tuning levers — real vs no-op (2026 research)

For bandwidth-bound OpenFOAM 13 + OGL + Ginkgo 2.0 (SYCL), 8 MPI ranks sharing one
B70. Triaged **REAL** vs **NO-OP** because each wasted rebuild/run costs ~1h.
Builds on: CR pinned 26.05 ([intel-compute-runtime-and-driver.md](intel-compute-runtime-and-driver.md)),
FP64 not the wall ([intel-platform-fit.md](intel-platform-fit.md)). All claims cited in the source research.

## ★ TOP-5 levers to try next (cheapest → highest payoff)
1. **`ranksPerGPU` (OGL repartitioning) + `matrixFormat Csr`** — fuses N CPU-rank
   matrices onto 1 GPU rank. Oversubscribing a GPU *without* this is catastrophic
   (KIT/Anzt [arXiv:2510.08536](https://arxiv.org/abs/2510.08536): 0.007× = ~140×
   slower; GPU needs ~1M DOFs/rank). **We already set `ranksPerGPU = np`** (8/8 → 1
   fused partition) → we're in the good regime; this explains why "more ranks
   helped." Worth confirming, not a new win — but verify it stays = np.
2. **`clpeak --transfer-bandwidth` + read the *parent-bridge* link** before chasing
   the 0.94 GB/s D2H. The "PCIe Gen1×1" lspci reading is an **Arc switch-hierarchy
   artifact**; real link is Gen5-class (~48–56 GB/s on B70). The slow D2H is a
   software transfer path (pageable host mem / tiny copies), not the link.
3. **Verify `executor sycl;`** (OGL default is `reference` = CPU!) + pin the B70 via
   `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (guards the OGL "all"-vs-"gpu" device-index
   bug → iGPU contamination). One line, zero cost.
4. **A/B `SYCL_UR_USE_LEVEL_ZERO_V2=0`** (drop to V1 → re-enable command *batching*;
   B70 defaults to V2 = immediate-only) ± `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE`.
   Env-only, same-session A/B. 8 contending ranks may prefer batched submission.
5. **Pin GPU clock** (`rp0_freq` → both `min_freq`/`max_freq` in
   `…/tile0/gt0/freq0/`) to kill ±2–3% jitter (makes A/Bs trustworthy) +
   `SYCL_CACHE_PERSISTENT=1` (one-time 8-rank JIT-startup win).

## Confirmed NO-OPs — do NOT spend rebuild cycles
- **CCS partitioning** (`ZEX_NUMBER_OF_CCS`) — multi-CCS is Xe-HPC (PVC) only; B70 is
  single-tile. Splitting EUs starves a bandwidth-bound solver.
- **Sub-device / multi-tile**: `ZE_AFFINITY_MASK` sub-device syntax, `ZE_FLAT_DEVICE_HIERARCHY`,
  `EnableImplicitScaling`, `EngineInstancedSubDevices` — all multi-tile (PVC/Max); B70 single-tile.
- **`EnableRecoverablePageFaults`** — "faultable hardware" = PVC/Max, not consumer B70.
- **GuC/HuC toggles** — mandatory-on in xe, no tunable; just verify dmesg loads firmware.
- **GPU-aware MPI** — its 25–50% gain is a multi-GPU number; on one B70 it's intra-device → our measured "no win" is expected. `forceHostBuffer` keeps it moot.

## Traps
- **`GINKGO_DPCPP_SINGLE_MODE=ON`** → double kernels become `GKO_NOT_IMPLEMENTED`
  (runtime-fatal), **not** a silent FP32 downcast. Keep OFF; our FP32-preconditioner
  patch is the correct path to FP32.
- **`GINKGO_JACOBI_FULL_OPTIMIZATIONS`** — CUDA-only benefit; on SYCL just flips an
  unroll pragma. Skip (inflates build time).

## Build flags worth considering (one-time, not steady-state)
- `GINKGO_MIXED_PRECISION=ON` — true mixed-precision kernels (fewer bytes moved) —
  aligns with our FP32/VRAM direction; MAYBE real.
- AOT: `-fsycl-targets=spir64_gen -Xs "-device bmg"` — kills JIT warmup (startup only).
- `-ffp-model=precise` — Ginkgo-recommended SYCL correctness; negligible perf cost
  (we're bandwidth- not FP-bound), can shift iteration counts otherwise.

## NEO debug keys (need `NEOReadDebugKeys=1` first; experimental, measure don't trust)
- `EnableDirectSubmission` (bypass KMD, lower submit latency) — MAYBE; two-edged with
  8 ranks (ring contention). `ContextGroupSize` (how N contexts map to engines) —
  MAYBE; under-documented. Both cheap env A/Bs, no rebuild. Intel's own stance:
  debug keys are experiments, defaults are meant to be optimal → expect marginal.

> Mechanism note (validates our thesis): SYCL SpMV already hits ~90% of peak
> bandwidth on Intel GPUs; the cost is the SYCL *preconditioner software* (emulated
> subgroup ops, SLM bank conflicts) + AMG setup — not FP64. AMG hierarchy reuse
> ([amg-reuse-port-plan.md](amg-reuse-port-plan.md)) stays #1; no flag substitutes.
