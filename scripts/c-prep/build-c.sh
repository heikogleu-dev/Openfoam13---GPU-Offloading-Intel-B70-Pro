#!/bin/bash
# Plan C build — run AFTER the port edits (see knowledge/amg-reuse-port-plan.md).
# Enables the OGL extension + rebuilds Ginkgo-SYCL + OGL. Expect ~1h (SYCL backend).
set -e
B=/opt/ogl-src/build/release
echo "[pre] Confirm port edits are in place:"
grep -q "class UpdateMatrixValue" "$B/_deps/ginkgo-src/include/ginkgo/core/multigrid/multigrid_level.hpp" \
  && echo "  ✓ UpdateMatrixValue interface present" || { echo "  ✗ interface MISSING — do the port first"; exit 1; }
grep -q "GINKGO_WITH_OGL_EXTENSIONS" /opt/ogl-src/include/OGL/Preconditioner.hpp \
  && echo "  ✓ OGL #ifdef uses plural EXTENSIONS" || echo "  ⚠ check OGL #ifdef is plural (...EXTENSIONS) at Preconditioner.hpp:667"
echo "[1/3] CMake reconfigure: GINKGO_WITH_OGL_EXTENSIONS=ON"
cmake -B "$B" -S /opt/ogl-src -DGINKGO_WITH_OGL_EXTENSIONS=ON >/dev/null
echo "[2/3] Build (ninja) — ~1h..."
time ninja -C "$B"
echo "[3/3] Artifacts:"
ls -la --time-style=+%H:%M "$B/libOGL.so" "$B/_deps/ginkgo-build/lib/libginkgo_dpcpp.so"
grep "GINKGO_WITH_OGL_EXTENSIONS" "$B/CMakeCache.txt"
echo "Done. If it failed to compile: iterate on the port (errors are the develop API drift)."
echo "Rollback: restore *.bak-20260618-prereuse in the ginkgo-src + OGL trees."
