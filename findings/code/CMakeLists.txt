cmake_minimum_required(VERSION 3.16)
project(ginkgo_precond_sweep CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the OGL-installed Ginkgo 2.0
set(Ginkgo_DIR "/home/heiko/OpenFOAM/heiko-13/platforms/linux64GccDPInt32Opt/lib/cmake/Ginkgo")
find_package(Ginkgo REQUIRED)

add_executable(ginkgo_precond_sweep ginkgo_precond_sweep.cpp)
target_link_libraries(ginkgo_precond_sweep PRIVATE Ginkgo::ginkgo)
target_compile_options(ginkgo_precond_sweep PRIVATE -fsycl)
target_link_options(ginkgo_precond_sweep PRIVATE -fsycl)
