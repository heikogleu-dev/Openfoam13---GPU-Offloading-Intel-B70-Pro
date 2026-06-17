#!/bin/bash
# ILU rank sweep on the 7.1M half-res mesh: for each np, decompose to np,
# set ranksPerGPU=np, run 3 steps, report steady pressure-iteration count.
# Tests whether the high iteration count is an additive-Schwarz effect
# (more ranks -> more block fragmentation -> more iterations).
CASE=/home/heiko/CFD-Cases/Testcase-half
cd "$CASE"
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1

run_np() {
  local np=$1
  echo "############################################################"
  echo "### np=$np  $(date +%H:%M:%S)"
  sed -i -E "s/numberOfSubdomains [0-9]+;/numberOfSubdomains $np;/" system/decomposeParDict
  sed -i -E "s/ranksPerGPU +[0-9]+;/ranksPerGPU         $np;/" system/fvSolution
  rm -rf processor* [1-9]* 2>/dev/null
  decomposePar -force > log.decomp-np$np 2>&1
  if [ ! -d processor0 ]; then echo "np=$np: decomposePar FEHLER"; tail -3 log.decomp-np$np; return; fi
  local LOG=log.ILU-np$np
  timeout 600 mpirun -np $np foamRun -parallel -solver incompressibleFluid > "$LOG" 2>&1
  # steady iteration counts (2nd timestep) + per-step time
  echo "--- np=$np: Druck-Iterationen pro Solve (alle Steps) ---"
  grep "Solving for p" "$LOG" 2>/dev/null | sed -E 's/.*No Iterations/iter:/' | tr '\n' ' '; echo
  echo "--- np=$np: ExecutionTime pro Step ---"
  grep "ExecutionTime" "$LOG" 2>/dev/null | awk '{print $3}' | tr '\n' ' '; echo
  grep -iE "FATAL|terminate|DEVICE_LOST|allocate memory|find_blocks|18446744" "$LOG" 2>/dev/null | head -2
}

for np in 1 2 4 8 12; do run_np $np; done
echo "############################################################"
echo "### SWEEP FERTIG $(date +%H:%M:%S)"
