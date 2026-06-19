#!/bin/bash
# Full-float performance map. Sequential GPU runs (one B70) over a spec list,
# capturing per-config: steady s/step, p-iters, init_precond(build/reuse), solve,
# COMM (all-to-all gather / pairwise / reorder / copy-back), GPU util (ccs/rcs/bcs),
# VRAM peak. Re-decomposes only when np changes (order specs by np to minimise).
#
# Spec file: one config per line "NP|LABEL|<preconditioner dict body>"
#   16|mg-single-c3|preconditioner Multigrid; coarseSolver CG; maxIterCoarse 20; precision single; caching 3;
#   16|bj-single   |preconditioner BJ; maxBlockSize 1; precision single;
# Lines starting with # ignored.
#
# Usage: sweep-fullfloat.sh <case> <specfile> [endTime=12] [utilWindow=75]
set -u
CASE=${1:?case}; SPECS=${2:?specfile}; ENDT=${3:-12}; WIN=${4:-75}
DIAG=/home/heiko/gpu-diag/build
CD=$CASE/system/controlDict; FV=$CASE/system/fvSolution; DP=$CASE/system/decomposeParDict
CSV=$CASE/sweep-results.csv
cd "$CASE" || exit 1
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1
ranks(){ for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = foamRun ]&&echo $p; done; }
snap(){ local sc=0 sr=0 sb=0 tot=0
  for p in $(ranks); do local f=$(ls /proc/$p/fdinfo/* 2>/dev/null|head -60)
    local mc=$(grep -h "drm-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    local mr=$(grep -h "drm-cycles-rcs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    local mb=$(grep -h "drm-cycles-bcs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    local mt=$(grep -h "drm-total-cycles-ccs:" $f 2>/dev/null|awk '{print $2}'|sort -n|tail -1)
    sc=$((sc+${mc:-0})); sr=$((sr+${mr:-0})); sb=$((sb+${mb:-0})); [ "${mt:-0}" -gt "$tot" ]&&tot=${mt:-0}; done
  echo "$sc $tot $sr $sb"; }
vram(){ local v=0; for p in $(ranks); do for f in /proc/$p/fdinfo/*; do
    local m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0" "$f" 2>/dev/null|awk '{for(i=1;i<=NF;i++)if($i~/^[0-9]+$/)s=$i}END{print s+0}'); v=$((v+${m:-0})); done; done; echo $v; }
# steady mean of an OGL timing label (drop first 6 solves = startup+build cycles)
tmean(){ grep -h "p: $1:" log.$2 2>/dev/null|awk '{print $(NF-1)}'|LC_ALL=C awk 'NR>6{s+=$1;n++}END{if(n)printf "%.1f",s/n; else print "-"}'; }
tmin(){  grep -h "p: $1:" log.$2 2>/dev/null|awk '{print $(NF-1)}'|sort -n|head -1; }
tmax(){  grep -h "p: $1:" log.$2 2>/dev/null|awk '{print $(NF-1)}'|sort -n|tail -1; }

sed -i -E 's/^startTime[[:space:]]+[0-9]+;/startTime       0;/' $CD
sed -i -E "s/^endTime[[:space:]]+[0-9]+;/endTime         $ENDT;/" $CD
grep -q libOGL $CD || sed -i -E 's/^(application[[:space:]]+foamRun;)/\1\nlibs            ("libOGL.so");/' $CD
echo "config,np,label,sstep,iters_mean,initp_reuse,initp_build,solve_ms,all2all_ms,pairwise_ms,reorder_ms,copyback_ms,ccs_pct,rcs_pct,bcs_pct,vram_MiB" > $CSV
LAST_NP=""
echo "################ FULL-FLOAT SWEEP: $CASE  endTime=$ENDT ################ $(date +%H:%M:%S)"
while IFS='|' read -r NP LABEL BODY; do
  [ -z "${NP// }" ] && continue; case "$NP" in \#*) continue;; esac
  NP=$(echo $NP|tr -d ' '); LABEL=$(echo $LABEL|tr -d ' '); BODY=$(echo "$BODY"|sed 's/^ *//;s/ *$//')
  # (re)decompose only if np changed AND not already partitioned for NP
  if [ "$NP" != "$LAST_NP" ]; then
    curnp=$(ls -d processor* 2>/dev/null | wc -l)
    curdef=$(grep -oE 'numberOfSubdomains [0-9]+' $DP | grep -oE '[0-9]+' | head -1)
    if [ "$curnp" = "$NP" ] && [ "$curdef" = "$NP" ] && [ -d processor0/0 ]; then
      echo "=== already decomposed np=$NP, reuse partition ($(date +%H:%M:%S)) ==="
      rm -rf processor*/[1-9]* processor*/0.* 2>/dev/null
    else
      echo "=== decompose np=$NP ($(date +%H:%M:%S)) ==="
      sed -i -E "s/numberOfSubdomains [0-9]+;/numberOfSubdomains $NP;/" $DP
      rm -rf processor* 2>/dev/null
      decomposePar -force < /dev/null > log.dec-$NP 2>&1
      [ -d processor0 ] || { echo "  decompose FAIL np=$NP"; LAST_NP=""; continue; }
    fi
    LAST_NP=$NP
  else
    rm -rf processor*/[1-9]* processor*/0.* 2>/dev/null
  fi
  # write p-solver block
  cat > /tmp/pblock.$$ <<EOF
solvers
{
    p
    {
        solver GKOCG; executor sycl; verbose 1; matrixFormat Csr;
        preconditioner { $BODY }
        forceHostBuffer true; ranksPerGPU $NP; tolerance 1e-6; relTol 0.01; maxIter 200; adaptMinIter true; relaxationFactor 0.8; updateInitGuess true;
    }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.1; maxIter 50; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
    omega { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }
}
EOF
  # replace solvers{...} block in fvSolution (keep SIMPLE/relax below)
  awk 'BEGIN{skip=0} /^solvers/{skip=1; while((getline l < "/tmp/pblock.'$$'")>0) print l; next} skip&&/^}/{skip=0; next} skip{next} {print}' $FV > /tmp/fv.$$ && mv /tmp/fv.$$ $FV
  echo ">>> [$LABEL] np=$NP : $BODY  ($(date +%H:%M:%S))"
  nohup mpirun -np $NP foamRun -parallel -solver incompressibleFluid < /dev/null > log.$LABEL 2>&1 &
  MPID=$!
  for i in $(seq 1 200); do [ -n "$(ranks|head -1)" ] && [ "$(grep -c 'Solving for p' log.$LABEL 2>/dev/null)" -ge 6 ]&&break; kill -0 $MPID 2>/dev/null||break; sleep 3; done
  peak=0; read Ac At Ar Ab <<< "$(snap)"; ta=$(date +%s)
  end=$(( $(date +%s)+WIN )); while [ "$(date +%s)" -lt "$end" ]; do v=$(vram); [ "$v" -gt "$peak" ]&&peak=$v; kill -0 $MPID 2>/dev/null||break; sleep 3; done
  read Bc Bt Br Bb <<< "$(snap)"
  la=$(date +%s); lsz=0
  while kill -0 $MPID 2>/dev/null; do v=$(vram); [ "$v" -gt "$peak" ]&&peak=$v
    csz=$(stat -c %s log.$LABEL 2>/dev/null||echo 0); now=$(date +%s)
    [ "$csz" != "$lsz" ] && { la=$now; lsz=$csz; }
    [ $((now-la)) -gt 300 ] && { echo "    !! HANG >5min [$LABEL], killing"; kill -9 $MPID 2>/dev/null; pkill -9 -f "foamRun -parallel" 2>/dev/null; sleep 5; break; }
    sleep 4
  done
  ss=$(grep ExecutionTime log.$LABEL|awk '{print $3}'|LC_ALL=C awk 'NR>3{d=$1-p;s+=d;n++}{p=$1}END{if(n)printf "%.1f",s/n; else print "-"}')
  it=$(grep "Solving for p" log.$LABEL|sed -E 's/.*No Iterations //'|LC_ALL=C awk 'NR>6{s+=$1;n++}END{if(n)printf "%.1f",s/n; else print "-"}')
  read U_c U_t U_r U_b <<< "$(echo $((Bc-Ac)) $((Bt-At)) $((Br-Ar)) $((Bb-Ab)))"
  pc=$(LC_ALL=C awk -v a=$U_c -v t=$U_t 'BEGIN{if(t>0)printf "%.1f",100*a/t; else print "-"}')
  pr=$(LC_ALL=C awk -v a=$U_r -v t=$U_t 'BEGIN{if(t>0)printf "%.1f",100*a/t; else print "-"}')
  pb=$(LC_ALL=C awk -v a=$U_b -v t=$U_t 'BEGIN{if(t>0)printf "%.1f",100*a/t; else print "-"}')
  echo "$LABEL,$NP,\"$BODY\",$ss,$it,$(tmin init_precond $LABEL),$(tmax init_precond $LABEL),$(tmean solve $LABEL),$(tmean perform_all_to_all_update $LABEL),$(tmean perform_pairwise_update $LABEL),$(tmean reorder_matrix_data $LABEL),$(tmean copy_x_back $LABEL),$pc,$pr,$pb,$((peak/1024))" >> $CSV
  printf "    s/step=%-5s iters=%-4s solve=%-6s all2all=%-6s copyback=%-5s | CCS=%-4s BCS=%-4s | VRAM=%sMiB\n" "$ss" "$it" "$(tmean solve $LABEL)" "$(tmean perform_all_to_all_update $LABEL)" "$(tmean copy_x_back $LABEL)" "$pc" "$pb" "$((peak/1024))"
  grep -iqE "FATAL|bad_alloc|DEVICE_LOST" log.$LABEL && echo "    !! ERROR in log.$LABEL"
done < "$SPECS"
echo "################ SWEEP DONE $(date +%H:%M:%S) — results: $CSV ################"
column -t -s, "$CSV"
timeout 40 "$DIAG/diag-l0" >/dev/null 2>&1 && echo "GPU OK" || echo "GPU DEGRADED"
