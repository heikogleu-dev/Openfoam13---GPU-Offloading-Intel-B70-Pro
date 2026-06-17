#!/bin/bash
# Launch the 34M ILU multi-rank run with VRAM peak sampling (fdinfo) and
# hang detection. Prints results + peak VRAM on exit.
CASE=/home/heiko/CFD-Cases/Testcase-GPU
LOG="$CASE/log.post-recovery/test-ILU.log"
cd "$CASE"
source /home/heiko/github/intel-arc-pro-b70-openfoam/scripts/cr2605-shell.sh >/dev/null 2>&1

nohup mpirun -np 8 foamRun -parallel -solver incompressibleFluid > "$LOG" 2>&1 &
PID=$!
echo "PID: $PID — ILU, 34M, np=8, gestartet $(date +%H:%M:%S)"

peak_kb=0; last_size=0; stall=0
sample_vram() {
    # sum drm-total-vram across all foamRun rank fdinfo (xe driver)
    local total=0
    for p in $(pgrep -f "foamRun -parallel" 2>/dev/null); do
        local m=$(grep -h -i "drm-total-vram0\|drm-resident-vram0\|drm-total-local0" /proc/$p/fdinfo/* 2>/dev/null \
                  | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){s=$i}} END{print s+0}')
        total=$((total + ${m:-0}))
    done
    echo $total
}
for i in $(seq 1 360); do   # max 30 min
  if ! kill -0 $PID 2>/dev/null; then echo "=== mpirun EXITED nach ~$((i*5))s ==="; break; fi
  v=$(sample_vram); [ "${v:-0}" -gt "$peak_kb" ] && peak_kb=$v
  sz=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "$sz" -eq "$last_size" ]; then stall=$((stall+1)); else stall=0; last_size=$sz; fi
  if [ "$stall" -ge 66 ]; then echo "=== HANG >5.5min, kill ==="; kill -9 $PID 2>/dev/null; break; fi
  sleep 5
done
echo "########## PEAK VRAM (fdinfo sum, alle Ranks): $((peak_kb/1024)) MiB ##########"
echo "########## ERGEBNIS ILU ##########"
grep -E "Selecting solver|Solving for p|Time = |ExecutionTime|No Iterations|Final residual|FOAM FATAL|terminate|what\(\)|DEVICE_LOST|MPI_ABORT|find_blocks|allocate memory block|lower_trs|NotImplemented|bad_alloc|out of|oom" "$LOG" 2>/dev/null | tail -50
