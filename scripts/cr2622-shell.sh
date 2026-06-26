#!/bin/bash
# Activate Intel Compute Runtime 26.22 + IGC 2.36.3 (LLVM 17) backend in this shell.
# Tests whether CR 26.22 fixes the multi-rank zeInit abort (issue #922) AND whether
# the newer IGC improves SYCL kernel perf. Non-invasive LD-switch (no sudo).
#   source scripts/cr2622-shell.sh ; mpirun -np 8 foamRun -parallel -solver incompressibleFluid
CR_DIR="${HOME}/intel-cr-26.22"
export LD_LIBRARY_PATH="${CR_DIR}/usr/lib/x86_64-linux-gnu:${CR_DIR}/usr/local/lib:${LD_LIBRARY_PATH}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
echo "[cr2622-shell] LD prepended: $(realpath ${CR_DIR}/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1 2>/dev/null) + IGC $(realpath ${CR_DIR}/usr/local/lib/libigc.so.2 2>/dev/null|grep -oE '2\.[0-9.]+' )"
