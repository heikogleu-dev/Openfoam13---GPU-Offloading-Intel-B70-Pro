// Multi-rank Ginkgo: each rank builds an N×N Poisson matrix and runs a CG
// solve with BJ preconditioner — mimics OGL's per-rank GPU workload (large
// matrix + iterative solve), WITHOUT OGL's distributed-matrix coupling.
//
// Isolates: "is the OGL 34M failure raw multi-rank GPU capacity/size, or
// OGL-distributed-specific?" If 8 ranks each solve a multi-GB matrix here
// with the CR 26.05 LD-switch, the GPU handles the load → OGL failure is
// in the distributed wrapper, not raw capacity.
//
// arg1 = N (grid side, rows = N*N), arg2 = max_iters

#include <mpi.h>
#include <ginkgo/ginkgo.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define LOG(...) do { fprintf(stderr, "[rank %d/%d] ", rank, size); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using ValueType = double;
    using IndexType = int;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    IndexType N = argc > 1 ? std::atoi(argv[1]) : 2000;  // 2000 -> 4M rows ~1.2 GB
    int max_iters = argc > 2 ? std::atoi(argv[2]) : 50;

    LOG("MPI init OK, N=%d (%d rows)", N, N*N);

    // Barrier before SYCL init (the workaround that helps minimal Ginkgo)
    MPI_Barrier(MPI_COMM_WORLD);

    std::shared_ptr<gko::Executor> exec;
    try {
        exec = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
        LOG("OK DpcppExecutor");
    } catch (std::exception& e) { LOG("FAIL exec: %s", e.what()); MPI_Abort(MPI_COMM_WORLD,1); }

    MPI_Barrier(MPI_COMM_WORLD);

    // Build Poisson matrix on THIS rank (full local matrix, no distribution)
    const IndexType n_rows = N * N;
    gko::matrix_data<ValueType, IndexType> md(gko::dim<2>{(unsigned long)n_rows,(unsigned long)n_rows});
    md.nonzeros.reserve(5*n_rows);
    for (IndexType j=0;j<N;++j) for (IndexType i=0;i<N;++i){
        IndexType r=j*N+i;
        md.nonzeros.push_back({r,r,4.0});
        if(i>0)md.nonzeros.push_back({r,r-1,-1.0});
        if(i<N-1)md.nonzeros.push_back({r,r+1,-1.0});
        if(j>0)md.nonzeros.push_back({r,r-N,-1.0});
        if(j<N-1)md.nonzeros.push_back({r,r+N,-1.0});
    }
    std::shared_ptr<mtx> A;
    try {
        A = gko::share(mtx::create(exec));
        A->read(md);
        exec->synchronize();
        LOG("OK matrix on GPU (%lu nnz)", (unsigned long)A->get_num_stored_elements());
    } catch (std::exception& e) { LOG("FAIL matrix: %s", e.what()); MPI_Abort(MPI_COMM_WORLD,2); }

    auto b = gko::share(vec::create(exec, gko::dim<2>{(unsigned long)n_rows,1}));
    b->fill(1.0);
    auto x = gko::share(vec::create(exec, gko::dim<2>{(unsigned long)n_rows,1}));
    x->fill(0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();
    try {
        auto solver = cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters((unsigned long)max_iters))
            .with_preconditioner(bj::build().with_max_block_size(1u))
            .on(exec)->generate(A);
        solver->apply(b, x);
        exec->synchronize();
    } catch (std::exception& e) { LOG("FAIL solve: %s", e.what()); MPI_Abort(MPI_COMM_WORLD,3); }
    auto t1 = std::chrono::steady_clock::now();
    LOG("OK CG solve %d iters in %.1f ms", max_iters,
        std::chrono::duration<double,std::milli>(t1-t0).count());

    MPI_Barrier(MPI_COMM_WORLD);
    LOG("ALL DONE");
    MPI_Finalize();
    return 0;
}
