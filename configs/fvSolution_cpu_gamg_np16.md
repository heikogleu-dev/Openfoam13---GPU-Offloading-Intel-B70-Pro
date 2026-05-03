# fvSolution: CPU GAMG np=16 (Best Overall — 35.7 s/Step)

```
solvers
{
    p
    {
        solver              GAMG;
        tolerance           1e-6;
        relTol              0.01;
        smoother            DICGaussSeidel;
        cacheAgglomeration  true;
        nCellsInCoarsestLevel 500;
        agglomerator        faceAreaPair;
        mergeLevels         1;
        maxIter             50;
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
    nNonOrthogonalCorrectors 2;
    consistent      yes;
}
```

## decomposeParDict

```
numberOfSubdomains 16;
method             scotch;
```

## Run

```bash
cd /path/to/case
decomposePar -force
mpirun -np 16 foamRun -parallel -solver incompressibleFluid > log.foamRun 2>&1
```

No taskset / explicit binding — let the kernel scheduler use both P and E
cores. Adding `taskset -c 0-7` to limit to P-cores only is **slower** for
this workload (43.3 s/step vs 35.7 s/step).

## Why This Wins

- GAMG (algebraic multigrid) is O(N) optimal for the discrete Poisson p-system
- E-Cores contribute meaningfully to the multigrid-smoother work (compute-bound part)
- RAM bandwidth is saturated at 8 P-Cores anyway, so the E-Cores don't compete
- No PCIe round-trips, no Level-Zero kernel launch latency overhead
