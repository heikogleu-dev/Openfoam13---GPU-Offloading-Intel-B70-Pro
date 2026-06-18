#!/bin/bash
# 18M performance MAP (Testcase-mid): (preconditioner x ranks) -> util, VRAM, iters, wall-clock.
#  - MG V+CG-coarse rank curve: np = 4,8,12,16
#  - preconditioner comparison at np=8: MG-default, MG-WcgC, ILU
#  - CPU GAMG baseline at np=8 and np=16
# Each run: peak VRAM (fdinfo), compute-engine util (drm-cycles-ccs), pressure iters, ExecutionTime.
CASE=${1:?Usage: compare-mesh.sh <case-dir>}
DIAG=/home/heiko/gpu-diag/build
cd "$CASE"
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1

ranks(){ for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ] && echo $p; done; }
health(){ timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1; }
decomp(){ sed -i -E "s/numberOfSubdomains [0-9]+;/numberOfSubdomains $1;/" system/decomposeParDict; rm -rf processor* 2>/dev/null; decomposePar -force > log.dec-$1 2>&1; [ -d processor0 ]; }

gen_gko(){ cat > system/fvSolution <<EOF
FoamFile { version 2.0; class dictionary; format ascii; object fvSolution; }
solvers
{
    p
    {
        solver GKOCG; executor sycl; verbose 2; matrixFormat Csr;
        preconditioner
        {
$1
        }
        forceHostBuffer true; ranksPerGPU $2; tolerance 1e-6; relTol 0.01; maxIter 200;
        adaptMinIter true; relaxationFactor 0.8; updateInitGuess true; evalFrequency 10;
    }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
}
SIMPLE { consistent true; nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } pRefPoint (0 3 3); pRefValue 0.0; }
relaxationFactors { fields { p 0.5; } equations { U 0.7; k 0.5; omega 0.5; } }
EOF
}
gen_gamg(){ cat > system/fvSolution <<'EOF'
FoamFile { version 2.0; class dictionary; format ascii; object fvSolution; }
solvers
{
    p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; nPreSweeps 1; nPostSweeps 2; nFinestSweeps 2; cacheAgglomeration true; nCellsInCoarsestLevel 100; agglomerator faceAreaPair; mergeLevels 1; maxIter 50; }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
}
SIMPLE { consistent true; nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } pRefPoint (0 3 3); pRefValue 0.0; }
relaxationFactors { fields { p 0.5; } equations { U 0.7; k 0.5; omega 0.5; } }
EOF
}

run(){ # $1 label  $2 np  $3 gpu(1/0)
  local LOG=log.18m-$1 np=$2 gpu=$3
  rm -rf processor*/[1-9]* 2>/dev/null
  nohup mpirun -np $np foamRun -parallel -solver incompressibleFluid > "$LOG" 2>&1 &
  local PID=$! peak=0 us=0 un=0 pc=0 ptot=0
  while kill -0 $PID 2>/dev/null; do
    if [ "$gpu" = 1 ]; then
      local v=0 cc=0 tot=0 r
      for r in $(ranks); do
        local f=$(ls /proc/$r/fdinfo/* 2>/dev/null)
        local m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" $f 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}')
        v=$((v+${m:-0}))
        local c=$(grep -h "drm-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
        local t=$(grep -h "drm-total-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
        cc=$((cc+${c:-0})); [ "${t:-0}" -gt "$tot" ] && tot=${t:-0}
      done
      [ "$v" -gt "$peak" ] && peak=$v
      if [ "$ptot" -gt 0 ] && [ "$tot" -gt "$ptot" ]; then
        local u=$(LC_ALL=C awk -v a=$((cc-pc)) -v t=$((tot-ptot)) 'BEGIN{printf "%.0f",100*a/t}'); us=$((us+u)); un=$((un+1)); fi
      pc=$cc; ptot=$tot
    fi
    sleep 4
  done
  local it tm util
  it=$(grep "Solving for p" "$LOG" 2>/dev/null|sed -E 's/.*No Iterations //'|tr '\n' ' ')
  tm=$(grep ExecutionTime "$LOG" 2>/dev/null|awk '{print $3}'|tr '\n' ' ')
  util=$([ $un -gt 0 ] && echo "$((us/un))%" || echo "-")
  printf "### %-14s np=%-2s | VRAM=%5sMiB util=%-4s | iters=[%s] | times=[%s]\n" "$1" "$np" "$((peak/1024))" "$util" "$it" "$tm"
  [ -z "$it" ] && { echo "    (kein Solve)"; grep -iE "terminate|FATAL|DEVICE_LOST|allocate|aborted" "$LOG" 2>/dev/null|head -1; }
}

MG="            preconditioner  Multigrid;\n            coarseSolver    CG;\n            maxIterCoarse   20;"
echo "########## MAP $(date +%H:%M:%S) ##########"
[ -d processor0 ] && echo "nCells: $(zcat processor0/constant/polyMesh/owner.gz 2>/dev/null|sed -n '1,20p'|grep -m1 note)"

for np in 2 4 8 12 16; do
  echo "===== decompose -> $np  $(date +%H:%M:%S) ====="
  decomp $np || { echo "decompose $np FAIL"; continue; }
  printf -v B "%b" "$MG"; gen_gko "$B" $np; run "MG-VcgC" $np 1; health||{ echo "  GPU degraded 30s"; sleep 30; }
  if [ "$np" = 8 ]; then
    gen_gko "            preconditioner  Multigrid;" $np; run "MG-default" $np 1; health||sleep 30
    printf -v B2 "%b" "            preconditioner  Multigrid;\n            cycle           w;\n            coarseSolver    CG;\n            maxIterCoarse   20;"; gen_gko "$B2" $np; run "MG-WcgC" $np 1; health||sleep 30
    gen_gko "            preconditioner  ILU;" $np; run "ILU" $np 1; health||{ echo "  GPU degraded 30s"; sleep 30; }
  fi
  if [ "$np" = 8 ] || [ "$np" = 16 ]; then
    gen_gamg; run "GAMGcpu" $np 0
  fi
done
# restore best
printf -v B "%b" "$MG"; gen_gko "$B" 8; decomp 8 >/dev/null 2>&1
echo "########## MAP FERTIG $(date +%H:%M:%S) ##########"