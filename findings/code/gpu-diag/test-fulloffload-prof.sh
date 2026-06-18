#!/bin/bash
# Full-offload test WITH full profiling: 17.2M np16 single. p-only vs (p+U+k+omega on GPU).
# Captures: VRAM peak, compute-util (ccs), copy/comm-util (bcs), per-phase timing (verbose 2).
CASE=/home/heiko/CFD-Cases/Testcase-mid
cd "$CASE"; source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1
cp -n system/fvSolution system/fvSolution.preFulloffloadTest 2>/dev/null
sed -i -E 's/^endTime         [0-9]+;/endTime         4;/' system/controlDict
ranks(){ for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ] && echo $p; done; }
P_GPU='    p { solver GKOCG; executor sycl; verbose 2; matrixFormat Csr; preconditioner { preconditioner Multigrid; coarseSolver CG; maxIterCoarse 20; precision single; } forceHostBuffer true; ranksPerGPU 16; tolerance 1e-6; relTol 0.01; maxIter 200; adaptMinIter true; relaxationFactor 0.8; updateInitGuess true; }'
U_CPU='    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }'
U_GPU='    U { solver GKOBiCGStab; executor sycl; verbose 2; matrixFormat Csr; preconditioner { preconditioner BJ; maxBlockSize 1; } forceHostBuffer true; ranksPerGPU 16; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver GKOBiCGStab; executor sycl; verbose 2; matrixFormat Csr; preconditioner { preconditioner BJ; maxBlockSize 1; } forceHostBuffer true; ranksPerGPU 16; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver GKOBiCGStab; executor sycl; verbose 2; matrixFormat Csr; preconditioner { preconditioner BJ; maxBlockSize 1; } forceHostBuffer true; ranksPerGPU 16; tolerance 1e-6; relTol 0.1; maxIter 50; }'
gen(){ cat > system/fvSolution <<EOF
FoamFile { version 2.0; class dictionary; format ascii; object fvSolution; }
solvers
{
$P_GPU
$1
}
SIMPLE { consistent true; nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } pRefPoint (0 3 3); pRefValue 0.0; }
relaxationFactors { fields { p 0.5; } equations { U 0.7; k 0.5; omega 0.5; } }
EOF
}
ph(){ grep -E "Proc: 0\].*: $2:" "$1" 2>/dev/null|grep -oE "$2: [0-9.]+"|awk '{s+=$2;n++}END{if(n)printf "%.0f(n=%d)",s/n,n;else printf "-"}'; }
run(){ gen "$2"; rm -rf processor*/[1-9]* 2>/dev/null; local L=log.FOP-$1
  timeout 700 mpirun -np 16 foamRun -parallel -solver incompressibleFluid > "$L" 2>&1 &
  local PID=$! peak=0 cs=0 cn=0 bs=0 bn=0 pcc=0 pct=0 pbc=0 pbt=0
  while kill -0 $PID 2>/dev/null; do
    local v=0 cc=0 ct=0 bc=0 bt=0 r
    for r in $(ranks); do local f=$(ls /proc/$r/fdinfo/* 2>/dev/null)
      local m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" $f 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}'); v=$((v+${m:-0}))
      local c=$(grep -h "drm-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1); cc=$((cc+${c:-0}))
      local ti=$(grep -h "drm-total-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1); [ "${ti:-0}" -gt "$ct" ]&&ct=${ti:-0}
      local b=$(grep -h "drm-cycles-bcs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1); bc=$((bc+${b:-0}))
      local tb=$(grep -h "drm-total-cycles-bcs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1); [ "${tb:-0}" -gt "$bt" ]&&bt=${tb:-0}
    done
    [ "$v" -gt "$peak" ]&&peak=$v
    [ "$pct" -gt 0 ]&&[ "$ct" -gt "$pct" ]&&{ u=$(LC_ALL=C awk -v a=$((cc-pcc)) -v t=$((ct-pct)) 'BEGIN{printf "%.0f",100*a/t}'); cs=$((cs+u)); cn=$((cn+1)); }
    [ "$pbt" -gt 0 ]&&[ "$bt" -gt "$pbt" ]&&{ u=$(LC_ALL=C awk -v a=$((bc-pbc)) -v t=$((bt-pbt)) 'BEGIN{printf "%.0f",100*a/t}'); bs=$((bs+u)); bn=$((bn+1)); }
    pcc=$cc; pct=$ct; pbc=$bc; pbt=$bt; sleep 3
  done
  local pit=$(grep "Solving for p" "$L"|sed -E 's/.*No Iterations //'|tr '\n' ' ')
  local cu=$([ $cn -gt 0 ]&&echo "$((cs/cn))%"||echo "-"); local bu=$([ $bn -gt 0 ]&&echo "$((bs/bn))%"||echo "-")
  local tm=$(grep ExecutionTime "$L"|awk '{print $3}'|tr '\n' ',')
  echo "### $1 | VRAM=$((peak/1024))MiB | compute-util(ccs)=$cu copy/comm-util(bcs)=$bu"
  echo "      PHASES-ms(alle Gleichungen): init_precond=$(ph "$L" init_precond) solve=$(ph "$L" solve) call_update=$(ph "$L" call_update) call_init=$(ph "$L" call_init) copy_x_back=$(ph "$L" copy_x_back)"
  echo "      p-iters=[$pit] | execT=[$tm]"
  grep -qiE "device lost|out of memory|GKO_NOT|terminate called|signal" "$L" && { echo "    ⚠ FEHLER:"; grep -iE "device lost|out of memory|GKO_NOT|terminate called|signal" "$L"|head -2; }
}
echo "########## FULL-OFFLOAD PROFILED 17.2M np16 single $(date +%H:%M:%S) ##########"
run "p-only"      "$U_CPU"
sleep 5
run "fulloffload" "$U_GPU"
cp -f system/fvSolution.preFulloffloadTest system/fvSolution 2>/dev/null
sed -i -E 's/^endTime         [0-9]+;/endTime         3;/' system/controlDict
echo "########## DONE $(date +%H:%M:%S) ##########"
