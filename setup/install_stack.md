# Installation Stack

## Prerequisites

```bash
# Intel GPU NEO Compute Runtime (from Ubuntu 26.04 universe — DO NOT add Intel
# graphics apt repo on resolute, it only ships up to noble and would conflict)
sudo apt install intel-opencl-icd libze1 libze-intel-gpu1 \
                 libze-intel-gpu-raytracing intel-ocloc clinfo intel-gpu-tools

# oneAPI 2026 (via Intel apt repo apt.repos.intel.com/oneapi all main)
sudo apt install intel-basekit
source /opt/intel/oneapi/setvars.sh

# OpenFOAM 13 Foundation
# resolute (26.04) is NOT in openfoam.org repo. Use questing (25.10) — it
# depends on libopenmpi40 which matches Ubuntu 26.04's OpenMPI 5.x.
echo 'deb [signed-by=/usr/share/keyrings/openfoam-archive-keyring.gpg] http://dl.openfoam.org/ubuntu questing main' | \
  sudo tee /etc/apt/sources.list.d/openfoam.list
sudo apt update && sudo apt install openfoam13
```

## Critical Build Fixes Required

### 1. icpx 2026 — removed `-fsycl-device-lib=all`

In OGL's internally-fetched Ginkgo CMakeLists, patch out this flag (the
build will fail at linking `libginkgo_dpcpp.so` otherwise):

```bash
# After first cmake configure (which fetches Ginkgo):
sed -i 's|target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)|# patched out: -fsycl-device-lib=all removed in icpx 2026|' \
  /opt/ogl-src/build/_deps/ginkgo-src/dpcpp/CMakeLists.txt
sed -i 's|-fsycl-device-lib=all -fsycl-device-code-split=per_kernel|-fsycl-device-code-split=per_kernel|' \
  /opt/ogl-src/build/_deps/ginkgo-src/cmake/create_test.cmake
```

The flag was the historical default in icpx and is now implicit.

### 2. OpenFOAM 13 API Changes

**`fvCFD.H` umbrella header removed in OF13** — create a shim:

```bash
mkdir -p /opt/ogl-src/include/foam-shim
cat > /opt/ogl-src/include/foam-shim/fvCFD.H <<'SHIM'
// fvCFD.H Shim for OpenFOAM 13 (Foundation) — header was removed in OF13.
// Re-bundles the 21 sub-headers the original fvCFD.H provided. Two old
// includes (gravityMeshObject.H, columnFvMesh.H) are gone in OF13 — not
// needed by OGL.
#pragma once
#include "Time.H"
#include "fvMesh.H"
#include "fvc.H"
#include "fvMatrices.H"
#include "fvm.H"
#include "linear.H"
#include "uniformDimensionedFields.H"
#include "calculatedFvPatchFields.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include "findRefCell.H"
#include "IOMRFZoneList.H"
#include "constants.H"
#include "OSspecific.H"
#include "argList.H"
#include "timeSelector.H"
#ifndef namespaceFoam
#define namespaceFoam
using namespace Foam;
#endif
SHIM
```

**`cyclicFvPatch::nbrPatchID()` renamed to `nbrPatchIndex()` in OF13:**

```bash
sed -i 's/patch.nbrPatchID()/patch.nbrPatchIndex()/g' \
  /opt/ogl-src/src/MatrixWrapper/HostMatrix.cpp
```

### 3. gcc-16 required for icpx C++ headers

Ubuntu 26.04 has gcc-15 default + gcc-16 alongside. icpx fails the cmake
C++ compiler test if it can't decide which gcc's libstdc++ to use. Install
gcc-16 + libstdc++-16-dev and pin icpx via flag:

```bash
sudo apt install libstdc++-16-dev gcc-16 g++-16
```

Then in cmake:
```
-DCMAKE_C_COMPILER=gcc-16 \
-DCMAKE_CXX_COMPILER=icpx \
-DCMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/16 -I/opt/ogl-src/include/foam-shim"
```

### 4. ONEAPI_DEVICE_SELECTOR Syntax

```bash
# CORRECT — selects level_zero device index 0 (the B70 Pro)
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# WRONG — crashes with "Too many colons (:)"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu:0
```

### 5. Round-Robin GPU Pitfall (Critical for Performance)

When iGPU is enabled, `sycl-ls` shows TWO Level-Zero devices:
- `level_zero:0` → Battlemage B70 Pro
- `level_zero:1` → Arrow Lake iGPU

Without explicit selector, OGL's `compute_device_id()` does round-robin
across BOTH devices. The 4 ranks landing on iGPU run extremely slowly →
~15× total performance loss.

**Always set `ONEAPI_DEVICE_SELECTOR=level_zero:0` for the B70 Pro.**

### 6. Foundation OF13 needs `OGL_USE_FOAM_FOUNDATION_VERSION` auto-detect

OGL auto-detects via existence of `$WM_PROJECT_DIR/META-INFO` (only in ESI's
.com version). For Foundation, no flag needed — auto-detected.

### 7. `make install` re-links binaries with install RPATH

`pkexec make install` fails with `undefined reference to __kmpc_*@VERSION`
because the root shell lacks oneAPI in `LD_LIBRARY_PATH`. Correct invocation:

```bash
pkexec bash -c '
  source /opt/intel/oneapi/setvars.sh --config=$HOME/.oneapi-no-mpi.cfg >/dev/null
  cd /opt/ogl-src/build && make install
'
```

## OGL CMake Build Command (working)

```bash
cd /opt/ogl-src/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-16 \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/16 -I/opt/ogl-src/include/foam-shim" \
  -DGINKGO_BUILD_SYCL=ON \
  -DGINKGO_BUILD_OMP=ON \
  -DGINKGO_BUILD_CUDA=OFF \
  -DGINKGO_BUILD_HIP=OFF \
  -DGINKGO_BUILD_TESTS=OFF
make -j$(nproc)
```

After successful build:
```bash
cp /opt/ogl-src/build/libOGL.so $FOAM_USER_LIBBIN/
```

In `system/controlDict` of every case using OGL solvers:
```
libs ("libOGL.so");
```

## oneAPI No-MPI Config (avoids OpenMPI/Intel MPI conflict)

oneAPI's full setvars.sh puts Intel MPI's `mpicc`/`mpirun` ahead of system
OpenMPI 5.x in PATH, which silently breaks OpenFOAM parallel runs (Foam is
linked against `libopenmpi40`).

`~/.oneapi-no-mpi.cfg`:
```
default=exclude
advisor=latest
ccl=latest
compiler=latest
dal=latest
debugger=latest
dev-utilities=latest
dnnl=latest
dpcpp-ct=latest
dpl=latest
ipp=latest
ippcp=latest
mkl=latest
tbb=latest
tcm=latest
umf=latest
vtune=latest
```

In `~/.bashrc`:
```bash
source /opt/intel/oneapi/setvars.sh --config=$HOME/.oneapi-no-mpi.cfg > /dev/null
source /opt/openfoam13/etc/bashrc 2> >(grep -vE "showme:link|missing operand|Try 'dirname" >&2)
```

The OpenFOAM source-line filter suppresses two cosmetic but harmless
warnings (`mpicc --showme:link` from OpenMPI 5 incompat in Foam config and
a `dirname: missing operand` from a cleanup loop).
