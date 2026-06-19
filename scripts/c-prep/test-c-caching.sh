#!/bin/bash
# Plan C smoke/validation — run AFTER build-c.sh succeeds.
# A/B caching 0 (baseline) vs caching 1/3 (value-refresh). Verify: init_precond DROPS
# on cache hits AND iters do NOT creep (stale reuse → 201-cap = FAIL). 7.1M single np=8.
CASE=/home/heiko/CFD-Cases/Testcase-half
cd "$CASE"; source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1
sed -i -E 's/numberOfSubdomains [0-9]+;/numberOfSubdomains 8;/' system/decomposeParDict
rm -rf processor* 2>/dev/null; decomposePar -force >/dev/null 2>&1
sed -i -E 's/^endTime         [0-9]+;/endTime         6;/' system/controlDict
gen(){ # $1 = caching value
cat > system/fvSolution <<EOF
FoamFile { version 2.0; class dictionary; format ascii; object fvSolution; }
solvers
{
    p {
        solver GKOCG; executor sycl; verbose 2; matrixFormat Csr;
        preconditioner { preconditioner Multigrid; coarseSolver CG; maxIterCoarse 20; precision single; caching $1; }
        forceHostBuffer true; ranksPerGPU 8; tolerance 1e-6; relTol 0.01; maxIter 200;
        adaptMinIter true; relaxationFactor 0.8; updateInitGuess true;
    }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
}
SIMPLE { consistent true; nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } pRefPoint (0 3 3); pRefValue 0.0; }
relaxationFactors { fields { p 0.5; } equations { U 0.7; k 0.5; omega 0.5; } }
EOF
}
ph(){ grep -E "Proc: 0\].*: $2:" "$1" 2>/dev/null|grep -oE "$2: [0-9.]+"|awk '{s+=$2;n++}END{if(n)printf "%.0f",s/n;else printf "-"}'; }
run(){ gen $1; rm -rf processor*/[1-9]* 2>/dev/null; local L=log.C-caching$1
  nohup mpirun -np 8 foamRun -parallel -solver incompressibleFluid > "$L" 2>&1
  local it=$(grep "Solving for p" "$L"|sed -E 's/.*No Iterations //'|tr '\n' ' ')
  local tm=$(grep ExecutionTime "$L"|awk '{print $3}'|tr '\n' ',')
  echo "### caching=$1 | init_precond=$(ph "$L" init_precond)ms solve=$(ph "$L" solve)ms | iters=[$it] | execT=[$tm]"
  echo "$it" | grep -q "201" && echo "    ⚠ ITER-CREEP (201 cap) → reuse is stale, lower caching N or fix the value-refill"
}
echo "########## C CACHING A/B 7.1M single np=8 ##########"
run 0    # baseline (full rebuild every solve)
run 1    # reuse 1 solve then rebuild
run 3    # reuse 3 solves
sed -i -E 's/^endTime         [0-9]+;/endTime         3;/' system/controlDict
echo "########## DONE — PASS = init_precond drops on caching>0 AND iters stay ~12-15 (no 201) ##########"
