// Multi-rank SYCL diagnostic - mirror of diag-mpi-l0 but through SYCL queue API
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>

#define LOG(...) do { fprintf(stderr, "[rank %d/%d] ", rank, size); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    LOG("MPI init OK");

    if (getenv("DIAG_BARRIER_BEFORE_INIT")) {
        MPI_Barrier(MPI_COMM_WORLD);
        LOG("after MPI_Barrier (pre-SYCL)");
    }
    if (const char* s = getenv("DIAG_STAGGER_MS")) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rank * atoi(s)));
        LOG("after stagger sleep");
    }

    sycl::device gpu;
    try {
        auto t0 = std::chrono::steady_clock::now();
        gpu = sycl::device(sycl::gpu_selector_v);
        auto t1 = std::chrono::steady_clock::now();
        LOG("OK sycl::device (%.1f ms): %s",
            std::chrono::duration<double, std::milli>(t1-t0).count(),
            gpu.get_info<sycl::info::device::name>().c_str());
    } catch (sycl::exception& e) {
        LOG("FAIL sycl::device: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    sycl::queue q;
    try {
        auto t0 = std::chrono::steady_clock::now();
        q = sycl::queue(gpu, sycl::property::queue::in_order());
        auto t1 = std::chrono::steady_clock::now();
        LOG("OK sycl::queue (%.1f ms)",
            std::chrono::duration<double, std::milli>(t1-t0).count());
    } catch (sycl::exception& e) {
        LOG("FAIL sycl::queue: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // 1 MiB alloc
    try {
        int* p = sycl::malloc_device<int>(1024*1024/sizeof(int), q);
        q.memset(p, 0, 1024*1024).wait();
        LOG("OK sycl::malloc_device 1MiB + memset");
        sycl::free(p, q);
    } catch (sycl::exception& e) {
        LOG("FAIL sycl 1MiB: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // 100 MiB
    try {
        int* p = sycl::malloc_device<int>(100UL*1024*1024/sizeof(int), q);
        q.memset(p, 0, 100UL*1024*1024).wait();
        LOG("OK sycl::malloc_device 100MiB + memset");
        sycl::free(p, q);
    } catch (sycl::exception& e) {
        LOG("FAIL sycl 100MiB: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // tiny kernel
    try {
        int* p = sycl::malloc_device<int>(256, q);
        q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) { p[i] = (int)i.get(0); }).wait();
        LOG("OK sycl::parallel_for");
        sycl::free(p, q);
    } catch (sycl::exception& e) {
        LOG("FAIL sycl::parallel_for: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    LOG("ALL DONE");
    MPI_Finalize();
    return 0;
}
