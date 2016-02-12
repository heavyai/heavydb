#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"
#include "RexVisitor.h"

ResultRows RelAlgExecutor::executeRelAlgSeq(const std::list<RaExecutionDesc>&, const CompilationOptions&) {
  CHECK(false);
  return ResultRows("", 0);
}

namespace {

class RexUsedInputsVisitor : public RexVisitor<std::unordered_set<unsigned>> {
 public:
  std::unordered_set<unsigned> visitInput(const RexInput* rex_input) const override { return {rex_input->getIndex()}; }

 protected:
  std::unordered_set<unsigned> aggregateResult(const std::unordered_set<unsigned>& aggregate,
                                               const std::unordered_set<unsigned>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

std::unordered_set<unsigned> get_used_inputs(const RelCompound* compound) {
  RexUsedInputsVisitor visitor;
  auto used_inputs = visitor.visit(compound->getFilterExpr());
  const auto sources_size = compound->getScalarSourcesSize();
  for (size_t i = 0; i < sources_size; ++i) {
    const auto source_inputs = visitor.visit(compound->getScalarSource(i));
    used_inputs.insert(source_inputs.begin(), source_inputs.end());
  }
  return used_inputs;
}

}  // namespace

ResultRows RelAlgExecutor::executeCompound(const RelCompound* compound, const CompilationOptions& co) {
  const auto used_inputs = get_used_inputs(compound);
  CHECK(!used_inputs.empty());
  CHECK(false);
  return ResultRows("", 0);
}
#endif  // HAVE_CALCITE
