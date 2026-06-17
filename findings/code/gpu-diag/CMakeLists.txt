cmake_minimum_required(VERSION 3.16)
project(gpu_diag CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# L0 binary (no SYCL, no Ginkgo)
add_executable(diag-l0 diag-l0.cpp)
target_link_libraries(diag-l0 PRIVATE ze_loader)

# SYCL binary (no Ginkgo)
add_executable(diag-sycl diag-sycl.cpp)
target_compile_options(diag-sycl PRIVATE -fsycl)
target_link_options(diag-sycl PRIVATE -fsycl)

# Ginkgo binary
set(Ginkgo_DIR "/home/heiko/OpenFOAM/heiko-13/platforms/linux64GccDPInt32Opt/lib/cmake/Ginkgo")
find_package(Ginkgo REQUIRED)
add_executable(diag-ginkgo diag-ginkgo.cpp)
target_link_libraries(diag-ginkgo PRIVATE Ginkgo::ginkgo)
target_compile_options(diag-ginkgo PRIVATE -fsycl)
target_link_options(diag-ginkgo PRIVATE -fsycl)

# Multi-rank diagnostics
find_package(MPI REQUIRED)
add_executable(diag-mpi-l0 diag-mpi-l0.cpp)
target_link_libraries(diag-mpi-l0 PRIVATE ze_loader MPI::MPI_CXX)

add_executable(diag-mpi-sycl diag-mpi-sycl.cpp)
target_compile_options(diag-mpi-sycl PRIVATE -fsycl)
target_link_options(diag-mpi-sycl PRIVATE -fsycl)
target_link_libraries(diag-mpi-sycl PRIVATE MPI::MPI_CXX)

add_executable(diag-mpi-ginkgo diag-mpi-ginkgo.cpp)
target_link_libraries(diag-mpi-ginkgo PRIVATE Ginkgo::ginkgo MPI::MPI_CXX)
target_compile_options(diag-mpi-ginkgo PRIVATE -fsycl)
target_link_options(diag-mpi-ginkgo PRIVATE -fsycl)

add_executable(diag-mpi-ginkgo-solve diag-mpi-ginkgo-solve.cpp)
target_link_libraries(diag-mpi-ginkgo-solve PRIVATE Ginkgo::ginkgo MPI::MPI_CXX)
target_compile_options(diag-mpi-ginkgo-solve PRIVATE -fsycl)
target_link_options(diag-mpi-ginkgo-solve PRIVATE -fsycl)
