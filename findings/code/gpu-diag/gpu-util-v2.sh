#!/bin/bash
# Robust B70 compute-util measurement via fdinfo (no root).
CASE=/home/heiko/CFD-Cases/Testcase-half
cd "$CASE"
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1
sed -i -E 's/^endTime         [0-9]+;/endTime         10;/' system/controlDict
rm -rf processor*/[1-9]* 2>/dev/null
nohup mpirun -np 8 foamRun -parallel -solver incompressibleFluid > log.util 2>&1 &
MPID=$!
ranks() { for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ] && echo $p; done; }

# wait until we're into the solve (a rank exists + has a device fd), then settle
for i in $(seq 1 40); do r=$(ranks | head -1); [ -n "$r" ] && grep -q "Solving for p" log.util 2>/dev/null && break; sleep 2; done
sleep 8   # into steady state

echo "=== DEBUG: one rank's engine cycle line ==="
r=$(ranks | head -1); echo "rank PID $r"
cat /proc/$r/fdinfo/* 2>/dev/null | grep -E "drm-(cycles|total-cycles|engine-capacity)-(ccs|rcs|bcs|vcs|vecs):" | sort -u

snap() { # prints "ccs total rcs bcs" summed across ranks (max-per-rank to avoid fd dup)
  local sc=0 sr=0 sb=0 tot=0
  for p in $(ranks); do
    local f=$(ls /proc/$p/fdinfo/* 2>/dev/null | head -50)
    local mc=$(grep -h "drm-cycles-ccs:" $f 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    local mr=$(grep -h "drm-cycles-rcs:" $f 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    local mb=$(grep -h "drm-cycles-bcs:" $f 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    local mt=$(grep -h "drm-total-cycles-ccs:" $f 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    sc=$((sc+${mc:-0})); sr=$((sr+${mr:-0})); sb=$((sb+${mb:-0})); [ "${mt:-0}" -gt "$tot" ] && tot=${mt:-0}
  done
  echo "$sc $tot $sr $sb"
}
read A_ccs A_tot A_rcs A_bcs <<< "$(snap)"
echo "=== snapshot A: ccs=$A_ccs total=$A_tot rcs=$A_rcs bcs=$A_bcs ==="
sleep 15
read B_ccs B_tot B_rcs B_bcs <<< "$(snap)"
echo "=== snapshot B: ccs=$B_ccs total=$B_tot rcs=$B_rcs bcs=$B_bcs ==="

LC_ALL=C awk -v dc=$((B_ccs-A_ccs)) -v dr=$((B_rcs-A_rcs)) -v db=$((B_bcs-A_bcs)) -v dt=$((B_tot-A_tot)) \
  'BEGIN{ if(dt>0){printf "### CCS(compute)= %.1f%%   RCS(render)= %.1f%%   BCS(copy)= %.1f%%   (over %d total-cycles)\n",100*dc/dt,100*dr/dt,100*db/dt,dt} else print "dt=0 (no counter movement)"}'

kill -0 $MPID 2>/dev/null && { wait $MPID 2>/dev/null; }
echo "=== s/step ==="; grep ExecutionTime log.util | awk '{print $3}' | tr '\n' ' '; echo
sed -i -E 's/^endTime         [0-9]+;/endTime         3;/' system/controlDict