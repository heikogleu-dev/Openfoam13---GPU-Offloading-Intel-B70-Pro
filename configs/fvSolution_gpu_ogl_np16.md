# fvSolution: GPU OGL np=16 (Best GPU — 30.6 s/Step, reduced quality)

> ⚠️ **Note:** This uses `nNonOrth=1` and `maxIter=80` — both reduced from
> the CPU-fair config. With matching settings (nNonOrth=2, maxIter=200)
> GPU OGL gives ~50 s/step instead of 30.6.

```
solvers
{
    p
    {
        solver              GKOCG;
        executor            sycl;
        matrixFormat        Csr;
        preconditioner
        {
            preconditioner  BJ;
            maxBlockSize    1;          // Only working value for 34M mesh
        }
        forceHostBuffer     true;
        ranksPerGPU         16;
        tolerance           1e-5;
        relTol              0.01;
        maxIter             80;
        adaptMinIter        true;
        relaxationFactor    0.8;
        updateInitGuess     true;
    }

    U
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.1;
    }

    "(k|omega)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;     // reduced from 2 to gain speed
    consistent      yes;
    pRefPoint       (0 3 3);
}

relaxationFactors
{
    equations
    {
        U       0.9;
        k       0.9;
        omega   0.9;
    }
}
```

## controlDict — load OGL library

In `system/controlDict`:
```
libs ("libOGL.so");
```

(libOGL.so must be in `$FOAM_USER_LIBBIN`)

## decomposeParDict

```
numberOfSubdomains 16;
method             scotch;
```

## Run

```bash
cd /path/to/case
decomposePar -force
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
mpirun -np 16 foamRun -parallel -solver incompressibleFluid > log.foamRun 2>&1
```

## Why this is "fastest" but unfair

- `nNonOrthogonalCorrectors 1` → SIMPLE does 2 p-solves per timestep instead of 3
- `maxIter 80` → CG caps earlier (still hits cap, doesn't converge to relTol)
- `tolerance 1e-5` → less strict absolute convergence

The same relaxations on CPU GAMG would cut ~30% off CPU time (estimated
24 s/step). The GPU has no algorithmic advantage — only forced quality
trade-offs make it look faster.
