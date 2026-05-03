# Bug: Ginkgo 1.10 SYCL Multigrid — Divergenz + DEVICE_LOST

## Symptom

```
MultigridsyclGKOCG: Solving for p, Initial = 1, Final = 1.192881, No Iterations 31
terminate called after throwing: sycl::_V1::exception
what(): level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

dmesg:
```
xe 0000:04:00.0: [drm] VM worker error: -12 ← -ENOMEM = GPU-VRAM OOM
```

## Setup

- Mesh: 34M cells, 8 MPI ranks (4.25M cells/rank)
- Preconditioner: Multigrid (PGM coarsening, MaxLevels=5)
- Hardware: Intel Arc Pro B70 Pro, 32 GB GDDR6

## Root Cause

Ginkgo's PGM (Parallel Graph Matching) coarsening builds a 5-level hierarchy:
- Level 0: full matrix ~270 MB/rank
- Each level: Csr clone + Restriction + Prolongation operators
- 5 levels × 8 ranks × ~3 operators = massive VRAM footprint
- GPU VM runs out during Pgm::generate_local() → Csr::clone() → raw_copy_to()

Stack trace confirms: `gko::multigrid::Pgm::generate_local()` →
`gko::matrix::Csr::apply_impl()` → `raw_copy_to()` → DEVICE_LOST

## Behavior

- Exactly 1 p-solve executed (diverged: Final 1.193 > Initial 1.0)
- Multigrid diverges even before OOM crash
- GPU recovered automatically after crash (no reboot needed)

## Verdict

OGL Multigrid on distributed 34M-cell mesh is not viable with Ginkgo 1.10 SYCL.
Both algorithmic failure (divergence) and memory failure (OOM) occur simultaneously.

## Potential Fix

Rebuild OGL against Ginkgo 2.0 which has improved distributed multigrid
(PGM coarsening, better SYCL memory management).
