#!/bin/bash
CASE=${1:?case}; DIAG=/home/heiko/gpu-diag/build
cd "$CASE"; source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1
ranks(){ for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ] && echo $p; done; }
gen(){ cat > system/fvSolution <<EOF
FoamFile { version 2.0; class dictionary; format ascii; object fvSolution; }
solvers { p { solver GKOCG; executor sycl; verbose 2; matrixFormat Csr;
  preconditioner { preconditioner Multigrid; coarseSolver CG; maxIterCoarse 20; precision single; }
  forceHostBuffer true; ranksPerGPU $1; tolerance 1e-6; relTol 0.01; maxIter 200; adaptMinIter true; relaxationFactor 0.8; updateInitGuess true; evalFrequency 10; }
  U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
  k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
  omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; } }
SIMPLE { consistent true; nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } pRefPoint (0 3 3); pRefValue 0.0; }
relaxationFactors { fields { p 0.5; } equations { U 0.7; k 0.5; omega 0.5; } }
EOF
}
echo "########## SINGLE-PRECISION MG-VcgC MAP: $CASE $(date +%H:%M:%S) ##########"
for np in 2 4 8 12 16; do
  sed -i -E "s/numberOfSubdomains [0-9]+;/numberOfSubdomains $np;/" system/decomposeParDict
  rm -rf processor* 2>/dev/null; decomposePar -force > log.dec-$np 2>&1; [ -d processor0 ]||{ echo "dec $np FAIL";continue; }
  gen $np; rm -rf processor*/[1-9]* 2>/dev/null
  nohup mpirun -np $np foamRun -parallel -solver incompressibleFluid > log.single-np$np 2>&1 &
  PID=$!; peak=0; us=0; un=0; pc=0; ptot=0
  while kill -0 $PID 2>/dev/null; do
    v=0; cc=0; tot=0
    for r in $(ranks); do f=$(ls /proc/$r/fdinfo/* 2>/dev/null)
      m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" $f 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}'); v=$((v+${m:-0}))
      c=$(grep -h "drm-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1); t=$(grep -h "drm-total-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
      cc=$((cc+${c:-0})); [ "${t:-0}" -gt "$tot" ]&&tot=${t:-0}; done
    [ "$v" -gt "$peak" ]&&peak=$v
    [ "$ptot" -gt 0 ]&&[ "$tot" -gt "$ptot" ]&&{ u=$(LC_ALL=C awk -v a=$((cc-pc)) -v t=$((tot-ptot)) 'BEGIN{printf "%.0f",100*a/t}'); us=$((us+u)); un=$((un+1)); }
    pc=$cc; ptot=$tot; sleep 4
  done
  it=$(grep "Solving for p" log.single-np$np|sed -E 's/.*No Iterations //'|tr '\n' ' '); tm=$(grep ExecutionTime log.single-np$np|awk '{print $3}'|tr '\n' ' ')
  util=$([ $un -gt 0 ]&&echo "$((us/un))%"||echo "-")
  printf "### single np=%-2s | VRAM=%5sMiB util=%-4s | iters=[%s] | times=[%s]\n" "$np" "$((peak/1024))" "$util" "$it" "$tm"
  [ -z "$it" ]&&grep -iE "FATAL|terminate|DEVICE_LOST|allocate" log.single-np$np|head -1
  timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1||echo "  GPU DEGRADED"
done
echo "########## FERTIG $(date +%H:%M:%S) ##########"
