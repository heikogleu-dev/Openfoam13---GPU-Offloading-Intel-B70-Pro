// Ginkgo 2.0 SYCL preconditioner sweep on Intel Arc Pro B70 Pro (BMG-G31)
// Single-rank, no MPI — isolates Ginkgo algorithm bugs from OGL/multi-rank-CR-26.18 issues.
//
// Tests on a 2D Poisson 5-point matrix of size N×N (variable):
//   - BJ(maxBlockSize=1)
//   - BJ(maxBlockSize=2)     ← finding 02 reproducer (1.10/1.11: size_t underflow)
//   - ILU (ParIlu + lower_trs via oneMKL trsm, PR #2023)
//   - ICT (ParIct)           ← finding 05 reproducer (1.10/1.11: add_candidates SIGABRT)
//   - ISAI sparsityPower=1   ← finding 05 reproducer (1.10/1.11: diverges)

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <ginkgo/ginkgo.hpp>

using ValueType = double;
using IndexType = int;
using mtx = gko::matrix::Csr<ValueType, IndexType>;
using vec = gko::matrix::Dense<ValueType>;
using cg = gko::solver::Cg<ValueType>;

// Build N×N 2D Poisson 5-point matrix (N² rows, ~5N² nnz)
auto build_poisson_2d(std::shared_ptr<gko::Executor> exec, IndexType N)
{
    const IndexType n_rows = N * N;
    gko::matrix_data<ValueType, IndexType> md(gko::dim<2>{(unsigned long)n_rows, (unsigned long)n_rows});
    md.nonzeros.reserve(5 * n_rows);
    for (IndexType j = 0; j < N; ++j) {
        for (IndexType i = 0; i < N; ++i) {
            const IndexType r = j * N + i;
            md.nonzeros.push_back({r, r, 4.0});
            if (i > 0) md.nonzeros.push_back({r, r - 1, -1.0});
            if (i < N - 1) md.nonzeros.push_back({r, r + 1, -1.0});
            if (j > 0) md.nonzeros.push_back({r, r - N, -1.0});
            if (j < N - 1) md.nonzeros.push_back({r, r + N, -1.0});
        }
    }
    auto A = gko::share(mtx::create(exec));
    A->read(md);
    return A;
}

struct TestResult {
    std::string name;
    bool built_ok = false;
    bool generated_ok = false;
    bool solved_ok = false;
    int iters = -1;
    double residual = -1.0;
    std::string error;
    double t_total_ms = 0;
};

TestResult test_precond(
    const std::string& name,
    std::shared_ptr<gko::Executor> exec,
    std::shared_ptr<mtx> A,
    std::shared_ptr<vec> b,
    std::function<std::shared_ptr<gko::LinOpFactory>(std::shared_ptr<gko::Executor>)> precond_factory_fn,
    int max_iters = 200)
{
    TestResult R;
    R.name = name;
    auto t0 = std::chrono::steady_clock::now();
    try {
        // Build CG with the given preconditioner
        auto precond_factory = precond_factory_fn(exec);
        R.built_ok = true;

        auto solver_factory = cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters((unsigned long)max_iters),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_baseline(gko::stop::mode::initial_resnorm)
                    .with_reduction_factor(1e-6))
            .with_preconditioner(precond_factory)
            .on(exec);

        auto solver = solver_factory->generate(A);
        R.generated_ok = true;

        auto x = vec::create(exec, gko::dim<2>{b->get_size()[0], 1});
        x->fill(0.0);

        solver->apply(b, x);
        exec->synchronize();
        R.solved_ok = true;

        // Get iter count via logger? For simplicity skip detailed iter — just confirm solve completed
        R.iters = max_iters; // placeholder
        R.residual = -1.0;
    }
    catch (const gko::NotImplemented& e) {
        R.error = std::string("NotImplemented: ") + e.what();
    }
    catch (const gko::AllocationError& e) {
        R.error = std::string("AllocationError: ") + e.what();
    }
    catch (const std::exception& e) {
        R.error = std::string("Exception: ") + e.what();
    }
    catch (...) {
        R.error = "Unknown exception";
    }
    auto t1 = std::chrono::steady_clock::now();
    R.t_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return R;
}

int main(int argc, char** argv)
{
    std::cout << gko::version_info::get() << std::endl;

    IndexType N = argc > 1 ? std::atoi(argv[1]) : 1000;
    const std::string exec_str = argc > 2 ? argv[2] : "dpcpp";
    std::cout << "=== Ginkgo SYCL Preconditioner Sweep on BMG-G31 ===\n";
    std::cout << "Mesh: " << N << "x" << N << " = " << (N*N) << " rows (~"
              << (5*N*N) << " nnz)\n";
    std::cout << "Executor: " << exec_str << "\n\n";

    std::shared_ptr<gko::Executor> exec;
    if (exec_str == "dpcpp") {
        exec = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
    } else {
        exec = gko::ReferenceExecutor::create();
    }

    std::cout << "Building Poisson matrix..." << std::flush;
    auto A = build_poisson_2d(exec, N);
    auto n_rows = A->get_size()[0];
    std::cout << " " << A->get_num_stored_elements() << " nnz\n";

    auto b = gko::share(vec::create(exec, gko::dim<2>{n_rows, 1}));
    b->fill(1.0);

    // Test matrix
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using parilu = gko::factorization::ParIlu<ValueType, IndexType>;
    using parict = gko::factorization::ParIct<ValueType, IndexType>;
    using ilu_pc = gko::preconditioner::Ilu<ValueType, false, IndexType>;
    using isai = gko::preconditioner::Isai<gko::preconditioner::isai_type::general, ValueType, IndexType>;

    std::vector<TestResult> results;

    results.push_back(test_precond("BJ(maxBlockSize=1)", exec, A, b,
        [](auto e) {
            return bj::build().with_max_block_size(1u).on(e);
        }));

    results.push_back(test_precond("BJ(maxBlockSize=2)", exec, A, b,
        [](auto e) {
            return bj::build().with_max_block_size(2u).on(e);
        }));

    results.push_back(test_precond("ILU (ParIlu factor + Ilu apply)", exec, A, b,
        [](auto e) {
            auto fac = parilu::build().on(e);
            return ilu_pc::build()
                .with_factorization(gko::share(std::move(fac)))
                .on(e);
        }));

    results.push_back(test_precond("ICT (ParIct factor + Ic apply)", exec, A, b,
        [](auto e) {
            auto fac = parict::build().on(e);
            return ilu_pc::build()
                .with_factorization(gko::share(std::move(fac)))
                .on(e);
        }, 200));

    results.push_back(test_precond("ISAI sparsityPower=1", exec, A, b,
        [](auto e) {
            return isai::build().with_sparsity_power(1).on(e);
        }));

    // Report
    std::cout << "\n=== RESULTS ===\n";
    for (auto& R : results) {
        std::cout << "  " << R.name << ":\n";
        std::cout << "    factory built:    " << (R.built_ok ? "yes" : "no") << "\n";
        std::cout << "    solver generated: " << (R.generated_ok ? "yes" : "no") << "\n";
        std::cout << "    solve completed:  " << (R.solved_ok ? "yes" : "no") << "\n";
        if (!R.error.empty()) {
            std::cout << "    error:            " << R.error.substr(0, 200) << "\n";
        }
        std::cout << "    total ms:         " << R.t_total_ms << "\n";
        std::cout << "\n";
    }
    return 0;
}
