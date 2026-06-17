#!/bin/bash
# Comprehensive VRAM-monitored sweep on the 7.1M half-res mesh.
# PART A: ILU rank sweep (np=2/4/8/12) - iters step1-3 + s/step + peak VRAM
# PART B: preconditioner comparison @ np=8 - ILU/ISAI/ICT/BJ8/Multigrid
# Each run: peak VRAM via fdinfo (foamRun ranks), GPU health gate between runs.
CASE=/home/heiko/CFD-Cases/Testcase-half
DIAG=/home/heiko/gpu-diag/build
cd "$CASE"
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1

gen_fvSolution() {  # $1 = inner precond block, $2 = ranksPerGPU
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

health_ok() { timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1 && return 0 || return 1; }

run_monitored() {  # $1 = log, $2 = np ; prints iters/time/vram
  local logf=$1 np=$2
  rm -rf processor*/[1-9]* 2>/dev/null
  timeout 700 mpirun -np $np foamRun -parallel -solver incompressibleFluid > "$logf" 2>&1 &
  local PID=$! peak=0 v m
  while kill -0 $PID 2>/dev/null; do
    v=0
    for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do
      m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" /proc/$p/fdinfo/* 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}')
      v=$((v+${m:-0}))
    done
    [ "$v" -gt "$peak" ] && peak=$v
    sleep 3
  done
  local iters times
  iters=$(grep "Solving for p" "$logf" 2>/dev/null | sed -E 's/.*No Iterations //' | tr '\n' ',' | sed 's/,$//')
  times=$(grep "ExecutionTime" "$logf" 2>/dev/null | awk '{print $3}' | tr '\n' ',' | sed 's/,$//')
  echo "  peakVRAM=$((peak/1024))MiB  iters=[$iters]  execTimes=[$times]"
  local err; err=$(grep -iE "terminate|find_blocks|18446744|DEVICE_LOST|allocate memory|bad_alloc|FATAL|aborted" "$logf" 2>/dev/null | head -1)
  [ -n "$err" ] && echo "  ERROR: $err"
  [ -z "$iters" ] && echo "  (kein p-Solve -> Crash vor Solver)"
}

decomp() { sed -i -E "s/numberOfSubdomains [0-9]+;/numberOfSubdomains $1;/" system/decomposeParDict; rm -rf processor* [1-9]* 2>/dev/null; decomposePar -force > log.dec-$1 2>&1; }

echo "################# PART A: ILU Rank-Sweep (VRAM) #################"
for np in 2 4 8 12; do
  echo "### ILU np=$np  $(date +%H:%M:%S)"
  gen_fvSolution "            preconditioner  ILU;" $np
  decomp $np
  [ -d processor0 ] || { echo "  decompose FEHLER"; continue; }
  run_monitored log.A-ILU-np$np $np
  health_ok && echo "  GPU: OK" || echo "  GPU: DEGRADED"
done

echo "################# PART B: Preconditioner @ np=8 (VRAM) #################"
decomp 8
for cfg in "ISAI|            preconditioner  ISAI;" \
           "ICT|            preconditioner  ICT;" \
           "BJ8|            preconditioner  BJ;\n            maxBlockSize    8;" \
           "Multigrid|            preconditioner  Multigrid;"; do
  name=${cfg%%|*}; block=${cfg#*|}
  echo "### $name np=8  $(date +%H:%M:%S)"
  printf -v blk "%b" "$block"   # interpret \n
  gen_fvSolution "$blk" 8
  health_ok || { echo "  GPU DEGRADED vor Lauf - 30s warten"; sleep 30; }
  run_monitored log.B-$name-np8 8
  health_ok && echo "  GPU: OK" || echo "  GPU: DEGRADED"
done
echo "################# SWEEP FERTIG $(date +%H:%M:%S) #################"
