// Ginkgo diagnostic - tests Ginkgo Executor + array + matrix on SYCL
// Exit codes:
//   0 = ALL OK
//   1 = OmpExecutor create
//   2 = DpcppExecutor create
//   3 = gko::array alloc
//   4 = Csr matrix construct/read
//   5 = SpMV apply
//   6 = BJ(1) generate
//   7 = BJ(1) apply

#include <ginkgo/ginkgo.hpp>
#include <cstdio>
#include <vector>

#define TRY(name, code) do { \
    try { code; fprintf(stderr, "[ OK ] %s\n", name); } \
    catch (std::exception& e) { fprintf(stderr, "[FAIL] %s: %s\n", name, e.what()); return _err_code; } \
    catch (...) { fprintf(stderr, "[FAIL] %s: unknown exception\n", name); return _err_code; } \
} while (0)

int main() {
    using ValueType = double;
    using IndexType = int;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    fprintf(stderr, "=== Ginkgo Diagnostic ===\n");
    fprintf(stderr, "[INFO] Ginkgo version: %d.%d.%d\n",
            gko::version_info::get().header_version.major,
            gko::version_info::get().header_version.minor,
            gko::version_info::get().header_version.patch);

    std::shared_ptr<gko::Executor> host;
    int _err_code = 1;
    TRY("gko::OmpExecutor::create", host = gko::OmpExecutor::create());

    std::shared_ptr<gko::Executor> exec;
    _err_code = 2;
    TRY("gko::DpcppExecutor::create", exec = gko::DpcppExecutor::create(0, host));

    _err_code = 3;
    TRY("gko::array<double>(exec, 1024)", { gko::array<double> a(exec, 1024); a.fill(1.0); exec->synchronize(); });

    // Tiny 100x100 Poisson 2D
    const int N = 100;
    const int n_rows = N * N;
    gko::matrix_data<ValueType, IndexType> md(gko::dim<2>{(unsigned long)n_rows, (unsigned long)n_rows});
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const int r = j*N + i;
        md.nonzeros.push_back({r,r,4.0});
        if (i>0)       md.nonzeros.push_back({r,r-1,-1.0});
        if (i<N-1)     md.nonzeros.push_back({r,r+1,-1.0});
        if (j>0)       md.nonzeros.push_back({r,r-N,-1.0});
        if (j<N-1)     md.nonzeros.push_back({r,r+N,-1.0});
    }
    auto A = gko::share(mtx::create(exec));
    _err_code = 4;
    TRY("mtx::read (Poisson 10k rows)", A->read(md));

    auto b = gko::share(vec::create(exec, gko::dim<2>{(unsigned long)n_rows, 1}));
    b->fill(1.0);
    auto x = gko::share(vec::create(exec, gko::dim<2>{(unsigned long)n_rows, 1}));
    x->fill(0.0);

    _err_code = 5;
    TRY("A->apply(b, x)  // SpMV", { A->apply(b, x); exec->synchronize(); });

    _err_code = 6;
    std::shared_ptr<gko::LinOp> bj1_precond;
    TRY("BJ(1)::build()->generate(A)", {
        auto fac = bj::build().with_max_block_size(1u).on(exec);
        bj1_precond = fac->generate(A);
        exec->synchronize();
    });

    _err_code = 7;
    TRY("BJ(1)->apply(b, x)", { bj1_precond->apply(b, x); exec->synchronize(); });

    fprintf(stderr, "[PASS] All Ginkgo tests passed\n");
    return 0;
}
