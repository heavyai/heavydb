#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"
#include "RexVisitor.h"

ExecutionResult RelAlgExecutor::executeRelAlgSeq(std::list<RaExecutionDesc>& exec_descs, const CompilationOptions& co) {
  for (auto& exec_desc : exec_descs) {
    const auto body = exec_desc.getBody();
    const auto compound = dynamic_cast<const RelCompound*>(body);
    if (compound) {
      exec_desc.setResult(executeCompound(compound, co));
      continue;
    }
    CHECK(false);
  }
  CHECK(!exec_descs.empty());
  return exec_descs.back().getResult();
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
  const auto filter_expr = compound->getFilterExpr();
  std::unordered_set<unsigned> used_inputs = filter_expr ? visitor.visit(filter_expr) : std::unordered_set<unsigned>{};
  const auto sources_size = compound->getScalarSourcesSize();
  for (size_t i = 0; i < sources_size; ++i) {
    const auto source_inputs = visitor.visit(compound->getScalarSource(i));
    used_inputs.insert(source_inputs.begin(), source_inputs.end());
  }
  return used_inputs;
}

std::pair<std::vector<ScanId>, std::list<ScanColDescriptor>> get_scan_info(const RelCompound* compound,
                                                                           const int rte_idx) {
  const auto used_inputs = get_used_inputs(compound);
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
  return {scan_ids, scan_cols};
}

std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(const RelCompound* compound,
                                                                      const Catalog_Namespace::Catalog& cat,
                                                                      const int rte_idx) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
    scalar_sources.push_back(translate_scalar_rex(compound->getScalarSource(i), rte_idx, cat));
  }
  return scalar_sources;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelCompound* compound,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (const auto group_idx : compound->getGroupIndices()) {
    groupby_exprs.push_back(scalar_sources[group_idx]);
  }
  return groupby_exprs;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_quals(const RelCompound* compound,
                                                           const Catalog_Namespace::Catalog& cat,
                                                           const int rte_idx) {
  const auto filter_rex = compound->getFilterExpr();
  const auto filter_expr = filter_rex ? translate_scalar_rex(filter_rex, rte_idx, cat) : nullptr;
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  if (filter_expr) {
    quals.push_back(filter_expr);
  }
  return quals;
}

std::vector<Analyzer::Expr*> translate_targets(std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
                                               const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
                                               const RelCompound* compound,
                                               const Catalog_Namespace::Catalog& cat,
                                               const int rte_idx) {
  std::vector<Analyzer::Expr*> target_exprs;
  for (size_t i = 0; i < compound->size(); ++i) {
    const auto target_rex = compound->getTargetExpr(i);
    const auto target_rex_agg = dynamic_cast<const RexAgg*>(target_rex);
    if (target_rex_agg) {
      const auto target_expr = translate_aggregate_rex(target_rex_agg, rte_idx, cat, scalar_sources);
      target_exprs_owned.push_back(target_expr);
      target_exprs.push_back(target_expr.get());
    }
  }
  return target_exprs;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeCompound(const RelCompound* compound, const CompilationOptions& co) {
  int rte_idx = 0;  // TODO(alex)
  std::vector<ScanId> scan_ids;
  std::list<ScanColDescriptor> scan_cols;
  std::tie(scan_ids, scan_cols) = get_scan_info(compound, rte_idx);
  const auto scalar_sources = translate_scalar_sources(compound, cat_, rte_idx);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals = translate_quals(compound, cat_, rte_idx);
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  const auto target_exprs = translate_targets(target_exprs_owned, scalar_sources, compound, cat_, rte_idx);
  Executor::RelAlgExecutionUnit rel_alg_exe_unit{
      scan_ids, scan_cols, {}, quals, {}, groupby_exprs, target_exprs, {}, 0};
  size_t max_groups_buffer_entry_guess{2048};
  int32_t error_code{0};
  std::vector<Analyzer::TargetMetaInfo> targets_meta;
  CHECK_EQ(compound->size(), target_exprs.size());
  for (size_t i = 0; i < compound->size(); ++i) {
    CHECK(target_exprs[i]);
    targets_meta.emplace_back(compound->getFieldName(i), target_exprs[i]->get_type_info());
  }
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->catalog_ = &cat_;
  return {executor_->executeWorkUnit(&error_code,
                                     max_groups_buffer_entry_guess,
                                     true,
                                     get_query_infos(scan_ids, cat_),
                                     rel_alg_exe_unit,
                                     co,
                                     {false, true, false, false},
                                     cat_,
                                     executor_->row_set_mem_owner_,
                                     nullptr),
          targets_meta};
}
#endif  // HAVE_CALCITE
