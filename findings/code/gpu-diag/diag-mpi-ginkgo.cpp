// Multi-rank Ginkgo diagnostic
#include <mpi.h>
#include <ginkgo/ginkgo.hpp>
#include <cstdio>
#include <cstdlib>

#define LOG(...) do { fprintf(stderr, "[rank %d/%d] ", rank, size); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    LOG("MPI init OK");

    if (getenv("DIAG_BARRIER_BEFORE_INIT")) {
        MPI_Barrier(MPI_COMM_WORLD);
        LOG("after MPI_Barrier (pre-Ginkgo)");
    }

    std::shared_ptr<gko::Executor> exec;
    try {
        auto host = gko::OmpExecutor::create();
        exec = gko::DpcppExecutor::create(0, host);
        LOG("OK DpcppExecutor::create");
    } catch (std::exception& e) {
        LOG("FAIL DpcppExecutor: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    try {
        gko::array<double> a(exec, 1024);
        a.fill(1.0);
        exec->synchronize();
        LOG("OK gko::array 1024 + fill");
    } catch (std::exception& e) {
        LOG("FAIL array: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    try {
        gko::array<double> big(exec, 25*1024*1024);  // 200 MiB
        big.fill(1.0);
        exec->synchronize();
        LOG("OK gko::array 200 MiB");
    } catch (std::exception& e) {
        LOG("FAIL big array: %s", e.what());
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    LOG("ALL DONE");
    MPI_Finalize();
    return 0;
}
