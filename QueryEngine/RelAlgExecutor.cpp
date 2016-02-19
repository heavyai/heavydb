#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"
#include "RexVisitor.h"

ResultRows RelAlgExecutor::executeRelAlgSeq(std::list<RaExecutionDesc>& exec_descs, const CompilationOptions& co) {
  for (auto& exec_desc : exec_descs) {
    const auto body = exec_desc.getBody();
    const auto compound = dynamic_cast<const RelCompound*>(body);
    if (compound) {
      exec_desc.setResult(executeCompound(compound, co));
      continue;
    }
    CHECK(false);
  }
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

ResultRows* RelAlgExecutor::executeCompound(const RelCompound* compound, const CompilationOptions& co) {
  const auto used_inputs = get_used_inputs(compound);
  CHECK(!used_inputs.empty());
  int rte_idx = 0;  // TODO(alex)
  std::vector<ScanId> scan_ids;
  std::list<ScanColDescriptor> scan_cols;
  {
    CHECK_EQ(size_t(1), compound->inputCount());
    const auto scan_ra = dynamic_cast<const RelScan*>(compound->getInput(0));
    CHECK(scan_ra);  // TODO(alex)
    scan_ids.emplace_back(scan_ra->getTableDescriptor()->tableId, rte_idx);
    for (const auto used_input : used_inputs) {
      scan_cols.emplace_back(used_input, scan_ra->getTableDescriptor(), rte_idx);
    }
  }
  const auto filter_expr = translate_rex(compound->getFilterExpr(), rte_idx, cat_);
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
    scalar_sources.push_back(translate_rex(compound->getScalarSource(i), rte_idx, cat_));
  }
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (const auto group_idx : compound->getGroupIndices()) {
    groupby_exprs.push_back(scalar_sources[group_idx]);
  }
  Executor::RelAlgExecutionUnit rel_alg_exe_unit{scan_ids, scan_cols, {}, {filter_expr}, {}, groupby_exprs, {}, {}, 0};
  size_t max_groups_buffer_entry_guess{2048};
  int32_t error_code{0};
  executor_->executeAggScanPlan(true,
                                {},
                                rel_alg_exe_unit,
                                co.hoist_literals_,
                                co.device_type_,
                                co.opt_level_,
                                cat_,
                                executor_->row_set_mem_owner_,
                                max_groups_buffer_entry_guess,
                                &error_code,
                                false,
                                true,
                                false,
                                false,
                                nullptr);
  CHECK(false);
  return new ResultRows("", 0);
}
#endif  // HAVE_CALCITE
