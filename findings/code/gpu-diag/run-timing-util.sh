#!/bin/bash
# Steady-state timing + GPU engine utilisation + VRAM for a GKO GPU solve.
# Util via fdinfo cycle deltas (ccs=compute, rcs=render, bcs=copy) over total-cycles,
# two snapshots WINDOW s apart once in the steady solve phase (averages GPU-active
# p-solve + CPU U/k/omega phases). Assumes the case is already configured for the
# GPU solver (GKOCG) with the libOGL libs entry in controlDict.
# Usage: run-timing-util.sh <case> <np> <endTime> [label] [window_s]
set -u
CASE=${1:?case}; NP=${2:?np}; ENDT=${3:?endTime}; LABEL=${4:-timing}; WIN=${5:-75}
DIAG=/home/heiko/gpu-diag/build
cd "$CASE" || exit 1
source "${CR_SHELL:-/home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh}" >/dev/null 2>&1
ranks(){ for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ] && echo $p; done; }
sed -i -E "s/^endTime[[:space:]]+[0-9]+;/endTime         $ENDT;/" system/controlDict
sed -i -E "s/^startTime[[:space:]]+[0-9]+;/startTime       0;/" system/controlDict
snap(){ local sc=0 sr=0 sb=0 tot=0
  for p in $(ranks); do local f=$(ls /proc/$p/fdinfo/* 2>/dev/null | head -60)
    local mc=$(grep -h "drm-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    local mr=$(grep -h "drm-cycles-rcs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    local mb=$(grep -h "drm-cycles-bcs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    local mt=$(grep -h "drm-total-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    sc=$((sc+${mc:-0})); sr=$((sr+${mr:-0})); sb=$((sb+${mb:-0})); [ "${mt:-0}" -gt "$tot" ]&&tot=${mt:-0}; done
  echo "$sc $tot $sr $sb"; }
vram(){ local v=0; for p in $(ranks); do for f in /proc/$p/fdinfo/*; do
    local m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" "$f" 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}'); v=$((v+${m:-0})); done; done; echo $v; }
echo "########## $LABEL: $CASE np=$NP endTime=$ENDT win=${WIN}s $(date +%H:%M:%S) ##########"
rm -rf processor*/[1-9]* processor*/0.* 2>/dev/null
nohup mpirun -np $NP foamRun -parallel -solver incompressibleFluid > log.$LABEL 2>&1 &
MPID=$!
# wait until past startup: into ~step 3 of the solve
for i in $(seq 1 120); do
  [ -n "$(ranks|head -1)" ] && [ "$(grep -c 'Solving for p' log.$LABEL 2>/dev/null)" -ge 6 ] && break
  kill -0 $MPID 2>/dev/null || { echo "  !! exited during startup"; break; }
  sleep 3
done
peak=0
read Ac At Ar Ab <<< "$(snap)"; ta=$(date +%s)
echo "  snapshot A: ccs=$Ac total=$At rcs=$Ar bcs=$Ab (t=$(date +%H:%M:%S))"
end=$(( $(date +%s) + WIN ))
while [ "$(date +%s)" -lt "$end" ]; do v=$(vram); [ "$v" -gt "$peak" ]&&peak=$v; sleep 3; kill -0 $MPID 2>/dev/null||break; done
read Bc Bt Br Bb <<< "$(snap)"; tb=$(date +%s)
echo "  snapshot B: ccs=$Bc total=$Bt rcs=$Br bcs=$Bb (t=$(date +%H:%M:%S))"
LC_ALL=C awk -v dc=$((Bc-Ac)) -v dr=$((Br-Ar)) -v db=$((Bb-Ab)) -v dt=$((Bt-At)) -v secs=$((tb-ta)) \
 'BEGIN{ if(dt>0) printf "### UTIL over %ds: CCS(compute)=%.1f%%  RCS(render)=%.1f%%  BCS(copy)=%.1f%%  (dt=%d cyc)\n",secs,100*dc/dt,100*dr/dt,100*db/dt,dt; else print "### UTIL: dt=0 (no counter movement)"}'
# let it finish for clean s/step
while kill -0 $MPID 2>/dev/null; do v=$(vram); [ "$v" -gt "$peak" ]&&peak=$v; sleep 4; done
echo "--- iters/p-solve ---"; grep "Solving for p" log.$LABEL|sed -E 's/.*No Iterations //'|tr '\n' ' '; echo
echo "--- ExecutionTime per step ---"; grep ExecutionTime log.$LABEL|awk '{print $3}'|tr '\n' ' '; echo
echo "--- steady s/step (consecutive ExecutionTime deltas, drop first 2 steps) ---"
grep ExecutionTime log.$LABEL|awk '{print $3}'|awk 'NR>1{d=$1-p; if(NR>3)printf "%.1f ",d}{p=$1}'; echo
echo "--- init_precond / solve (ms) sample ---"; grep -E "init_precond:|: solve:" log.$LABEL|tail -8
printf "### %s | VRAM peak=%s MiB (%.2f GB) ###\n" "$LABEL" "$((peak/1024))" "$(awk "BEGIN{print $peak/1048576}")"
grep -iE "FATAL|bad_alloc|DEVICE_LOST|out of memory" log.$LABEL|head -3
timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1 && echo "GPU OK" || echo "  GPU DEGRADED"
echo "########## DONE $(date +%H:%M:%S) ##########"
