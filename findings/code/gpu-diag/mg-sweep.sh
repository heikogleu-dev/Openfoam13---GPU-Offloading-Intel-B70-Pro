#!/bin/bash
# Multigrid tuning map on the 7.1M half-res mesh, np=8.
# Goal: how close can a tuned Ginkgo Multigrid get to CPU GAMG (7.7 s/step, 3-5 iter)?
# Default MG (V/Jacobi/1-sweep) = 55-101 iter, ~14 s/step, 10.4 GB.
CASE=/home/heiko/CFD-Cases/Testcase-half
DIAG=/home/heiko/gpu-diag/build
cd "$CASE"
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1

gen_fvSolution() {  # $1 = inner precond block (12-space indented), $2 = ranksPerGPU
  cat > system/fvSolution <<EOF
FoamFile { version 2.0; class dictionary; format ascii; object fvSolution; }
solvers
{
    p
    {
        solver              GKOCG;
        executor            sycl;
        verbose             2;
        matrixFormat        Csr;
        preconditioner
        {
$1
        }
        forceHostBuffer     true;
        ranksPerGPU         $2;
        tolerance           1e-6;
        relTol              0.01;
        maxIter             200;
        adaptMinIter        true;
        relaxationFactor    0.8;
        updateInitGuess     true;
        evalFrequency       10;
    }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
}
SIMPLE { consistent true; nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } pRefPoint (0 3 3); pRefValue 0.0; }
relaxationFactors { fields { p 0.5; } equations { U 0.7; k 0.5; omega 0.5; } }
EOF
}
health_ok() { timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1; }
run_monitored() {  # $1 log
  rm -rf processor*/[1-9]* 2>/dev/null
  timeout 700 mpirun -np 8 foamRun -parallel -solver incompressibleFluid > "$1" 2>&1 &
  local PID=$! peak=0 v m
  while kill -0 $PID 2>/dev/null; do
    v=0; for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do
      m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" /proc/$p/fdinfo/* 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}'); v=$((v+${m:-0})); done
    [ "$v" -gt "$peak" ] && peak=$v; sleep 3
  done
  local it tm; it=$(grep "Solving for p" "$1" 2>/dev/null|sed -E 's/.*No Iterations //'|tr '\n' ','|sed 's/,$//')
  tm=$(grep ExecutionTime "$1" 2>/dev/null|awk '{print $3}'|tr '\n' ','|sed 's/,$//')
  echo "  peakVRAM=$((peak/1024))MiB  iters=[$it]  execTimes=[$tm]"
  [ -z "$it" ] && { echo "  (kein p-Solve -> Crash)"; grep -iE "terminate|FATAL|DEVICE_LOST|aborted|what\(\)" "$1" 2>/dev/null|head -1; }
}

# fresh np=8 decomposition
sed -i -E 's/numberOfSubdomains [0-9]+;/numberOfSubdomains 8;/' system/decomposeParDict
rm -rf processor* [1-9]* 2>/dev/null; decomposePar -force > log.dec-mg 2>&1
[ -d processor0 ] || { echo "decompose FEHLER"; exit 1; }

P="            preconditioner  Multigrid;"
declare -a CFG=(
  "01-default|$P"
  "02-wcycle|$P\n            cycle           w;"
  "03-fcycle|$P\n            cycle           f;"
  "04-smooth3|$P\n            maxIterSmoother 3;"
  "05-ssor2|$P\n            smoother        SSOR;\n            maxIterSmoother 2;"
  "06-cgcoarse|$P\n            coarseSolver    CG;\n            maxIterCoarse   20;"
  "07-combo|$P\n            cycle           w;\n            smoother        SSOR;\n            maxIterSmoother 2;\n            coarseSolver    CG;\n            maxIterCoarse   20;"
  "08-deepcoarse|$P\n            minCoarseRows   8000;\n            coarseSolver    CG;\n            maxIterCoarse   30;\n            maxIterSmoother 2;"
)
for entry in "${CFG[@]}"; do
  name=${entry%%|*}; blk=${entry#*|}
  echo "############### MG $name  $(date +%H:%M:%S) ###############"
  printf -v B "%b" "$blk"
  gen_fvSolution "$B" 8
  health_ok || { echo "  GPU degraded - 30s warten"; sleep 30; }
  run_monitored "log.MG-$name"
  health_ok && echo "  GPU: OK" || echo "  GPU: DEGRADED"
done
echo "############### MG-SWEEP FERTIG $(date +%H:%M:%S) ###############"
