// Reference: gko::UpdateMatrixValue interface (Ginkgo 1.11 hpsim tree).
// Add to include/ginkgo/core/multigrid/multigrid_level.hpp, after `namespace gko {`.
namespace gko {
class UpdateMatrixValue {
public:
    virtual void update_matrix_value(std::shared_ptr<const gko::LinOp>) = 0;
};
}  // namespace gko
