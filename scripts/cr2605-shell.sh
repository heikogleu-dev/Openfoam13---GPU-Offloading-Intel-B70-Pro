#!/bin/bash
# Activate Intel Compute Runtime 26.05 Intel-GPU backend in this shell
# (workaround for the CR 26.18 multi-rank pthread_once race on BMG-G31,
# see findings/27).
#
# Usage:
#   source scripts/cr2605-shell.sh
#   mpirun -np 8 foamRun -parallel -solver incompressibleFluid

CR2605_DIR="${HOME}/intel-cr-26.05"

if [[ ! -d "${CR2605_DIR}/usr/lib/x86_64-linux-gnu" ]]; then
    echo "ERROR: CR 26.05 not found at ${CR2605_DIR}." >&2
    echo "" >&2
    echo "First-time setup (no sudo required):" >&2
    echo "  mkdir -p /tmp/cr26.05-download && cd /tmp/cr26.05-download" >&2
    echo "  apt download intel-opencl-icd=26.05.37020.3-1 \\" >&2
    echo "    libze-intel-gpu1=26.05.37020.3-1 intel-ocloc=26.05.37020.3-1" >&2
    echo "  mkdir -p ${CR2605_DIR}" >&2
    echo "  for d in *26.05*.deb; do dpkg-deb -x \"\$d\" ${CR2605_DIR}/; done" >&2
    return 1 2>/dev/null || exit 1
fi

# Prepend CR 26.05 Intel-GPU backend ahead of the system CR 26.18
export LD_LIBRARY_PATH="${CR2605_DIR}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"

echo "[cr2605-shell] LD_LIBRARY_PATH prepended with ${CR2605_DIR}/usr/lib/x86_64-linux-gnu"
echo "[cr2605-shell] Active Intel-GPU L0 backend: $(ldconfig -p 2>/dev/null | head -0 ; \
    realpath ${CR2605_DIR}/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1 2>/dev/null)"
