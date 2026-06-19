#!/bin/bash
# Full-float (plan D) single-run monitor: VRAM peak + GKO p-iters + s/step.
# Uses the case's EXISTING fvSolution (keep precision/caching as configured).
# Usage: run-fullfloat.sh <case> <np> [label]
set -u
CASE=${1:?case}; NP=${2:?np}; LABEL=${3:-run}; DIAG=/home/heiko/gpu-diag/build
cd "$CASE" || exit 1
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1
ranks(){ for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ] && echo $p; done; }
echo "########## $LABEL: $CASE np=$NP $(date +%H:%M:%S) ##########"
rm -rf processor*/[1-9]* processor*/0.* 2>/dev/null
nohup mpirun -np $NP foamRun -parallel -solver incompressibleFluid > log.$LABEL 2>&1 &
PID=$!; peak=0; last_activity=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  v=0
  for r in $(ranks); do
    for f in /proc/$r/fdinfo/*; do
      m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" "$f" 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}')
      v=$((v+${m:-0}))
    done
  done
  [ "$v" -gt "$peak" ] && peak=$v
  # hang guard: 5 min without log growth -> report (iron rule)
  now=$(date +%s); sz=$(stat -c %s log.$LABEL 2>/dev/null||echo 0)
  [ "${sz:-0}" != "${lastsz:-x}" ] && { last_activity=$now; lastsz=$sz; }
  [ $((now-last_activity)) -gt 300 ] && { echo "  !! HANG >5min, killing"; kill -9 $PID; break; }
  sleep 3
done
echo "--- p-solve GKO iters per solve ---"
grep "Solving for p" log.$LABEL | sed -E 's/.*Initial residual = ([0-9.e+-]+).*Final residual = ([0-9.e+-]+).*No Iterations ([0-9]+).*/init=\1 final=\2 iters=\3/' | head -40
echo "--- iters list ---"; grep "Solving for p" log.$LABEL | sed -E 's/.*No Iterations //' | tr '\n' ' '; echo
echo "--- ExecutionTime per step ---"; grep ExecutionTime log.$LABEL | awk '{print $3}' | tr '\n' ' '; echo
echo "--- init_precond / generate_solver / solve timings (ms) ---"
grep -E "init_precond:|generate_solver:|: solve:" log.$LABEL | tail -12
printf "### %s np=%-2s | VRAM peak=%s MiB (%.2f GB) ###\n" "$LABEL" "$NP" "$((peak/1024))" "$(awk "BEGIN{print $peak/1048576}")"
grep -iE "FATAL|terminate|DEVICE_LOST|allocate|bad_alloc|out of memory|signal" log.$LABEL | head -5
timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1 && echo "GPU OK" || echo "  GPU DEGRADED"
echo "########## DONE $(date +%H:%M:%S) ##########"
