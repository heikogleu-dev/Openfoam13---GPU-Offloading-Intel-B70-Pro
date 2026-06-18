// Reference: Pgm<ValueType,IndexType>::update_matrix_value (Ginkgo 1.11, core/multigrid/pgm.cpp:499).
// develop adaptation: distributed branch must reuse cached coarse_imap/off_diag_map (see knowledge/amg-reuse-port-plan.md).

void Pgm<ValueType, IndexType>::update_matrix_value(
    std::shared_ptr<const LinOp> new_matrix)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    system_matrix_ = new_matrix;
#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<
            const experimental::distributed::DistributedBase>(system_matrix_)) {
        auto convert_fine_op = [&](auto matrix) {
            using global_index_type = typename std::decay_t<
                decltype(*matrix)>::result_type::global_index_type;
            auto exec = as<LinOp>(matrix)->get_executor();
            auto comm = as<experimental::distributed::DistributedBase>(matrix)
                            ->get_communicator();
            auto fine = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        matrix::Csr<ValueType, IndexType>::create(exec),
                        matrix::Csr<ValueType, IndexType>::create(exec)));
            matrix->convert_to(fine);
            this->set_fine_op(fine);
        };
        auto setup_fine_op = [&](auto matrix) {
            // Only support csr matrix currently.
            auto local_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_local_matrix());
            auto non_local_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_non_local_matrix());
            // If system matrix is not csr or need sorting, generate the csr.
            if (!parameters_.skip_sorting || !local_csr || !non_local_csr) {
                using global_index_type =
                    typename std::decay_t<decltype(*matrix)>::global_index_type;
                convert_fine_op(
                    as<ConvertibleTo<experimental::distributed::Matrix<
                        ValueType, IndexType, global_index_type>>>(matrix));
            }
        };

        using fst_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
        using snd_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, int64>;
        // setup the fine op using Csr with current ValueType
        // we do not use dispatcher run in the first place because we have the
        // fallback option for that.
        if (auto obj =
                std::dynamic_pointer_cast<const fst_mtx_type>(system_matrix_)) {
            setup_fine_op(obj);
        } else if (auto obj = std::dynamic_pointer_cast<const snd_mtx_type>(
                       system_matrix_)) {
            setup_fine_op(obj);
        } else {
            // handle other ValueTypes.
            run<ConvertibleTo, fst_mtx_type, snd_mtx_type>(obj,
                                                           convert_fine_op);
        }

        auto distributed_setup = [&](auto matrix) {
            auto exec = gko::as<LinOp>(matrix)->get_executor();
            auto comm =
                gko::as<experimental::distributed::DistributedBase>(matrix)
                    ->get_communicator();
            auto num_rank = comm.size();
            auto pgm_local_op =
                gko::as<const csr_type>(matrix->get_local_matrix());

            auto num_agg =
                static_cast<IndexType>(this->get_coarse_op()->get_size()[0]);
            auto coarse_local_matrix = generate_coarse(
                this->get_executor(), pgm_local_op.get(), num_agg, agg_);

            auto non_local_csr =
                as<const csr_type>(matrix->get_non_local_matrix());
            auto result_non_local_csr = generate_coarse(
                exec, non_local_csr.get(), num_agg, agg_,
                static_cast<IndexType>(
                    matrix->get_non_local_matrix()->get_size()[1]),
                non_local_map_);

            // setup the generated linop.
            using global_index_type =
                typename std::decay_t<decltype(*matrix)>::global_index_type;
            // setup the generated linop.
            auto coarse = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, matrix->get_index_map(),
                        coarse_local_matrix, result_non_local_csr));
            this->set_multigrid_level(this->get_prolong_op(), coarse,
                                      this->get_restrict_op());
        };
        // the fine op is using csr with the current ValueType
        run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(), distributed_setup);
    } else
#endif
    {
        auto pgm_op = std::dynamic_pointer_cast<const csr_type>(system_matrix_);
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !pgm_op) {
            pgm_op = convert_to_with_sorting<csr_type>(
                this->get_executor(), system_matrix_, parameters_.skip_sorting);
            // keep the same precision data in fine_op
            this->set_fine_op(pgm_op);
        }
        auto coarse_matrix = generate_coarse(
            this->get_executor(), pgm_op.get(),
            static_cast<IndexType>(this->get_coarse_op()->get_size()[0]), agg_);
        this->set_multigrid_level(this->get_prolong_op(), coarse_matrix,
                                  this->get_restrict_op());
    }
}
