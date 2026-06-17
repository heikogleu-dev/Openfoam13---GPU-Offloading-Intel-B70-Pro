// Multi-rank Level Zero diagnostic
// Tests zeInit / driver / device / context / memAlloc on each MPI rank
// Run with: mpirun -np N ./diag-mpi-l0
// ENV options:
//   DIAG_BARRIER_BEFORE_INIT=1 — barrier before zeInit (serialize)
//   DIAG_STAGGER_MS=50 — sleep rank*N ms before zeInit (staggered init)

#include <mpi.h>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>

#define LOG(...) do { fprintf(stderr, "[rank %d/%d] ", rank, size); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LOG("MPI init OK");

    // Optional barrier before zeInit
    if (getenv("DIAG_BARRIER_BEFORE_INIT")) {
        MPI_Barrier(MPI_COMM_WORLD);
        LOG("after MPI_Barrier (pre-zeInit)");
    }

    // Optional stagger
    if (const char* s = getenv("DIAG_STAGGER_MS")) {
        int ms = atoi(s);
        std::this_thread::sleep_for(std::chrono::milliseconds(rank * ms));
        LOG("after stagger sleep %d ms", rank * ms);
    }

    auto t0 = std::chrono::steady_clock::now();
    ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    auto t1 = std::chrono::steady_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r != ZE_RESULT_SUCCESS) {
        LOG("FAIL zeInit: 0x%x (%.1f ms)", r, init_ms);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    LOG("OK zeInit (%.1f ms)", init_ms);

    // Driver / device
    uint32_t dc = 0;
    if (zeDriverGet(&dc, nullptr) != ZE_RESULT_SUCCESS || dc == 0) {
        LOG("FAIL zeDriverGet count=%u", dc);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::vector<ze_driver_handle_t> drivers(dc);
    zeDriverGet(&dc, drivers.data());

    uint32_t devc = 0;
    zeDeviceGet(drivers[0], &devc, nullptr);
    std::vector<ze_device_handle_t> devices(devc);
    zeDeviceGet(drivers[0], &devc, devices.data());
    LOG("OK driver(%u) device(%u)", dc, devc);

    // Context
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    if (zeContextCreate(drivers[0], &cd, &ctx) != ZE_RESULT_SUCCESS) {
        LOG("FAIL zeContextCreate");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    LOG("OK zeContextCreate");

    // Memory alloc 1 MiB
    void* ptr = nullptr;
    ze_device_mem_alloc_desc_t ad = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    r = zeMemAllocDevice(ctx, &ad, 1024*1024, 0, devices[0], &ptr);
    if (r != ZE_RESULT_SUCCESS) {
        LOG("FAIL zeMemAllocDevice 1MiB: 0x%x", r);
        zeContextDestroy(ctx);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    LOG("OK zeMemAllocDevice 1MiB at %p", ptr);
    zeMemFree(ctx, ptr);

    // Memory alloc 100 MiB
    r = zeMemAllocDevice(ctx, &ad, 100UL*1024*1024, 0, devices[0], &ptr);
    if (r != ZE_RESULT_SUCCESS) {
        LOG("FAIL zeMemAllocDevice 100MiB: 0x%x", r);
        zeContextDestroy(ctx);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    LOG("OK zeMemAllocDevice 100MiB");
    zeMemFree(ctx, ptr);

    zeContextDestroy(ctx);

    MPI_Barrier(MPI_COMM_WORLD);
    LOG("ALL DONE");
    MPI_Finalize();
    return 0;
}
