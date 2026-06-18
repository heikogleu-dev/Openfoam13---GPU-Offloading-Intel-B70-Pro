// Reference: Multigrid::update_matrix_value (Ginkgo 1.11, core/solver/multigrid.cpp:871).
// develop: ports nearly verbatim (set_system_matrix via EnableSolverBase, handle_list/smoother lists present).

void Multigrid::update_matrix_value(std::shared_ptr<const LinOp> new_matrix)
{
    this->set_system_matrix(new_matrix);
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = this->get_system_matrix()->get_size()[0];
    size_type level = 0;
    auto matrix = this->get_system_matrix();
    auto exec = this->get_executor();
    // clean all smoother
    pre_smoother_list_.clear();
    mid_smoother_list_.clear();
    post_smoother_list_.clear();
    // Always generate smoother with size = level.
    for (int i = 0; i < mg_level_list_.size(); i++) {
        auto mg_level = mg_level_list_.at(i);
        as<UpdateMatrixValue>(mg_level)->update_matrix_value(matrix);
        matrix = mg_level->get_coarse_op();
        run<gko::multigrid::EnableMultigridLevel, float, double,
            std::complex<float>, std::complex<double>>(
            mg_level,
            [this](auto mg_level, auto index, auto matrix) {
                using value_type =
                    typename std::decay_t<decltype(*mg_level)>::value_type;
                handle_list<value_type>(
                    index, matrix, parameters_.pre_smoother, pre_smoother_list_,
                    parameters_.smoother_iters, parameters_.smoother_relax);
                if (parameters_.mid_case ==
                    multigrid::mid_smooth_type::standalone) {
                    handle_list<value_type>(
                        index, matrix, parameters_.mid_smoother,
                        mid_smoother_list_, parameters_.smoother_iters,
                        parameters_.smoother_relax);
                }
                if (!parameters_.post_uses_pre) {
                    handle_list<value_type>(
                        index, matrix, parameters_.post_smoother,
                        post_smoother_list_, parameters_.smoother_iters,
                        parameters_.smoother_relax);
                }
            },
            i, mg_level->get_fine_op());
    }
    if (parameters_.post_uses_pre) {
        post_smoother_list_ = pre_smoother_list_;
    }
    auto last_mg_level = mg_level_list_.back();

    // generate coarsest solver
    run<gko::multigrid::EnableMultigridLevel, float, double,
        std::complex<float>, std::complex<double>>(
        last_mg_level,
        [this](auto mg_level, auto level, auto matrix) {
            using value_type =
                typename std::decay_t<decltype(*mg_level)>::value_type;
            auto exec = this->get_executor();
            // default coarse grid solver, direct LU
            // TODO: maybe remove fixed index type
            auto gen_default_solver = [&]() -> std::unique_ptr<LinOp> {
        // TODO: unify when dpcpp supports direct solver
#if GINKGO_BUILD_MPI
                if (gko::detail::is_distributed(matrix.get())) {
                    using absolute_value_type = remove_complex<value_type>;
                    using experimental::distributed::Matrix;
                    return run<Matrix<value_type, int32, int32>,
                               Matrix<value_type, int32, int64>,
                               Matrix<value_type, int64,
                                      int64>>(matrix, [exec](auto matrix) {
                        using Mtx = typename decltype(matrix)::element_type;
                        return solver::Gmres<value_type>::build()
                            .with_criteria(
                                stop::Iteration::build().with_max_iters(
                                    matrix->get_size()[0]),
                                stop::ResidualNorm<value_type>::build()
                                    .with_reduction_factor(
                                        std::numeric_limits<
                                            absolute_value_type>::epsilon() *
                                        absolute_value_type{10}))
                            .with_krylov_dim(
                                std::min(size_type(100), matrix->get_size()[0]))
                            .with_preconditioner(
                                experimental::distributed::preconditioner::
                                    Schwarz<value_type,
                                            typename Mtx::local_index_type,
                                            typename Mtx::global_index_type>::
                                        build()
                                            .with_local_solver(
                                                preconditioner::Jacobi<
                                                    value_type>::build()
                                                    .with_max_block_size(1u)))
                            .on(exec)
                            ->generate(matrix);
                    });
                }
#endif
                if (dynamic_cast<const DpcppExecutor*>(exec.get())) {
                    using absolute_value_type = remove_complex<value_type>;
                    return solver::Gmres<value_type>::build()
                        .with_criteria(
                            stop::Iteration::build().with_max_iters(
                                matrix->get_size()[0]),
                            stop::ResidualNorm<value_type>::build()
                                .with_reduction_factor(
                                    std::numeric_limits<
                                        absolute_value_type>::epsilon() *
                                    absolute_value_type{10}))
                        .with_krylov_dim(
                            std::min(size_type(100), matrix->get_size()[0]))
                        .with_preconditioner(
                            preconditioner::Jacobi<value_type>::build()
                                .with_max_block_size(1u))
                        .on(exec)
                        ->generate(matrix);
                } else {
                    return experimental::solver::Direct<value_type,
                                                        int32>::build()
                        .with_factorization(
                            experimental::factorization::Lu<value_type,
                                                            int32>::build())
                        .on(exec)
                        ->generate(matrix);
                }
            };
            if (parameters_.coarsest_solver.size() == 0) {
                coarsest_solver_ = gen_default_solver();
            } else {
                auto temp_index = solver_selector_(level, matrix.get());
                GKO_ENSURE_IN_BOUNDS(temp_index,
                                     parameters_.coarsest_solver.size());
                auto solver = parameters_.coarsest_solver.at(temp_index);
                if (solver == nullptr) {
