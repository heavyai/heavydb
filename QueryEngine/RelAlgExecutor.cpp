#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"

#include "InputMetadata.h"
#include "RexVisitor.h"

ExecutionResult RelAlgExecutor::executeRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs,
                                                 const CompilationOptions& co) {
  decltype(temporary_tables_)().swap(temporary_tables_);
  std::vector<TargetMetaInfo> in_metainfo;
  for (auto& exec_desc : exec_descs) {
    const auto body = exec_desc.getBody();
    const auto compound = dynamic_cast<const RelCompound*>(body);
    if (compound) {
      exec_desc.setResult(executeCompound(compound, in_metainfo, co));
      addTemporaryTable(-compound->getId(), &exec_desc.getResult().getRows());
      in_metainfo = exec_desc.getResult().getTargetsMeta();
      continue;
    }
    const auto project = dynamic_cast<const RelProject*>(body);
    if (project) {
      exec_desc.setResult(executeProject(project, in_metainfo, co));
      addTemporaryTable(-project->getId(), &exec_desc.getResult().getRows());
      in_metainfo = exec_desc.getResult().getTargetsMeta();
      continue;
    }
    const auto filter = dynamic_cast<const RelFilter*>(body);
    if (filter) {
      exec_desc.setResult(executeFilter(filter, in_metainfo, co));
      addTemporaryTable(-filter->getId(), &exec_desc.getResult().getRows());
      in_metainfo = exec_desc.getResult().getTargetsMeta();
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

std::unordered_set<unsigned> get_used_inputs(const RelProject* project) {
  RexUsedInputsVisitor visitor;
  std::unordered_set<unsigned> used_inputs;
  for (size_t i = 0; i < project->size(); ++i) {
    const auto proj_inputs = visitor.visit(project->getProjectAt(i));
    used_inputs.insert(proj_inputs.begin(), proj_inputs.end());
  }
  return used_inputs;
}

int table_id_from_ra(const RelAlgNode* ra_node) {
  const auto scan_ra = dynamic_cast<const RelScan*>(ra_node);
  if (scan_ra) {
    const auto td = scan_ra->getTableDescriptor();
    CHECK(td);
    return td->tableId;
  }
  return -ra_node->getId();
}

template <class RA>
std::pair<std::vector<InputDescriptor>, std::list<InputColDescriptor>>
get_input_desc_impl(const RA* ra_node, const int rte_idx, const std::unordered_set<unsigned>& used_inputs) {
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  CHECK_EQ(size_t(1), ra_node->inputCount());
  const auto input_ra = ra_node->getInput(0);
  const auto join_input = dynamic_cast<const RelJoin*>(input_ra);
  if (join_input) {
    CHECK_EQ(size_t(2), join_input->inputCount());
    const auto outer_ra = join_input->getInput(0);
    input_descs.emplace_back(table_id_from_ra(outer_ra), 0);
    const auto inner_ra = join_input->getInput(1);
    input_descs.emplace_back(table_id_from_ra(inner_ra), 1);
    CHECK(false);
  } else {
    const int table_id = table_id_from_ra(input_ra);
    input_descs.emplace_back(table_id, rte_idx);
    const auto scan_ra = dynamic_cast<const RelScan*>(input_ra);
    for (const auto used_input : used_inputs) {
      // Physical columns from a scan node are numbered from 1 in our system.
      input_col_descs.emplace_back(scan_ra ? used_input + 1 : used_input, table_id, rte_idx);
    }
  }
  return {input_descs, input_col_descs};
}

template <class RA>
std::pair<std::vector<InputDescriptor>, std::list<InputColDescriptor>> get_input_desc(const RA* ra_node,
                                                                                      const int rte_idx) {
  const auto used_inputs = get_used_inputs(ra_node);
  return get_input_desc_impl(ra_node, rte_idx, used_inputs);
}

size_t get_scalar_sources_size(const RelCompound* compound) {
  return compound->getScalarSourcesSize();
}

size_t get_scalar_sources_size(const RelProject* project) {
  return project->size();
}

const RexScalar* scalar_at(const size_t i, const RelCompound* compound) {
  return compound->getScalarSource(i);
}

const RexScalar* scalar_at(const size_t i, const RelProject* project) {
  return project->getProjectAt(i);
}

template <class RA>
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(const RA* ra_node,
                                                                      const Catalog_Namespace::Catalog& cat,
                                                                      const std::vector<TargetMetaInfo>& in_metainfo,
                                                                      const int rte_idx) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < get_scalar_sources_size(ra_node); ++i) {
    scalar_sources.push_back(translate_scalar_rex(scalar_at(i, ra_node), rte_idx, cat, in_metainfo));
  }
  return scalar_sources;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelCompound* compound,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  if (!compound->isAggregate()) {
    return {nullptr};
  }
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (const auto group_idx : compound->getGroupIndices()) {
    groupby_exprs.push_back(scalar_sources[group_idx]);
  }
  return groupby_exprs;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_quals(const RelCompound* compound,
                                                           const Catalog_Namespace::Catalog& cat,
                                                           const std::vector<TargetMetaInfo>& in_metainfo,
                                                           const int rte_idx) {
  const auto filter_rex = compound->getFilterExpr();
  const auto filter_expr = filter_rex ? translate_scalar_rex(filter_rex, rte_idx, cat, in_metainfo) : nullptr;
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
                                               const std::vector<TargetMetaInfo>& in_metainfo,
                                               const int rte_idx) {
  std::vector<Analyzer::Expr*> target_exprs;
  for (size_t i = 0; i < compound->size(); ++i) {
    const auto target_rex = compound->getTargetExpr(i);
    const auto target_rex_agg = dynamic_cast<const RexAgg*>(target_rex);
    std::shared_ptr<Analyzer::Expr> target_expr;
    if (target_rex_agg) {
      target_expr = translate_aggregate_rex(target_rex_agg, rte_idx, cat, scalar_sources);
    } else {
      const auto target_rex_scalar = dynamic_cast<const RexScalar*>(target_rex);
      target_expr = translate_scalar_rex(target_rex_scalar, rte_idx, cat, in_metainfo);
    }
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

// TODO(alex): Adjust interfaces downstream and make this not needed.
std::vector<Analyzer::Expr*> get_exprs_not_owned(const std::vector<std::shared_ptr<Analyzer::Expr>>& exprs) {
  std::vector<Analyzer::Expr*> exprs_not_owned;
  for (const auto expr : exprs) {
    exprs_not_owned.push_back(expr.get());
  }
  return exprs_not_owned;
}

template <class RA>
std::vector<TargetMetaInfo> get_targets_meta(const RA* ra_node, const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    targets_meta.emplace_back(ra_node->getFieldName(i), target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeCompound(const RelCompound* compound,
                                                const std::vector<TargetMetaInfo>& in_metainfo,
                                                const CompilationOptions& co) {
  int rte_idx = 0;  // TODO(alex)
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  std::tie(input_descs, input_col_descs) = get_input_desc(compound, rte_idx);
  const auto scalar_sources = translate_scalar_sources(compound, cat_, in_metainfo, rte_idx);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals = translate_quals(compound, cat_, in_metainfo, rte_idx);
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  const auto target_exprs = translate_targets(target_exprs_owned, scalar_sources, compound, cat_, in_metainfo, rte_idx);
  CHECK_EQ(compound->size(), target_exprs.size());
  Executor::RelAlgExecutionUnit rel_alg_exe_unit{
      input_descs, input_col_descs, {}, quals, {}, groupby_exprs, target_exprs, {}, 0};
  const auto targets_meta = get_targets_meta(compound, target_exprs);
  return executeWorkUnit(rel_alg_exe_unit, input_descs, targets_meta, compound->isAggregate(), co);
}

ExecutionResult RelAlgExecutor::executeProject(const RelProject* project,
                                               const std::vector<TargetMetaInfo>& in_metainfo,
                                               const CompilationOptions& co) {
  int rte_idx = 0;  // TODO(alex)
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  std::tie(input_descs, input_col_descs) = get_input_desc(project, rte_idx);
  const auto target_exprs_owned = translate_scalar_sources(project, cat_, in_metainfo, rte_idx);
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  Executor::RelAlgExecutionUnit rel_alg_exe_unit{
      input_descs, input_col_descs, {}, {}, {}, {nullptr}, target_exprs, {}, 0};
  const auto targets_meta = get_targets_meta(project, target_exprs);
  return executeWorkUnit(rel_alg_exe_unit, input_descs, targets_meta, false, co);
}

namespace {

std::vector<std::shared_ptr<Analyzer::Expr>> synthesize_inputs(const RelAlgNode* ra_node,
                                                               const std::vector<TargetMetaInfo>& in_metainfo,
                                                               const int rte_idx) {
  CHECK_EQ(size_t(1), ra_node->inputCount());
  const auto input = ra_node->getInput(0);
  const int table_id = table_id_from_ra(input);
  std::vector<std::shared_ptr<Analyzer::Expr>> inputs;
  const auto scan_ra = dynamic_cast<const RelScan*>(input);
  int input_idx = 0;
  for (const auto& input_meta : in_metainfo) {
    inputs.push_back(std::make_shared<Analyzer::ColumnVar>(
        input_meta.get_type_info(), table_id, scan_ra ? input_idx + 1 : input_idx, rte_idx));
    ++input_idx;
  }
  return inputs;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeFilter(const RelFilter* filter,
                                              const std::vector<TargetMetaInfo>& in_metainfo,
                                              const CompilationOptions& co) {
  int rte_idx = 0;  // TODO(alex)
  CHECK_EQ(size_t(1), filter->inputCount());
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  std::unordered_set<unsigned> used_inputs;
  for (size_t i = 0; i < in_metainfo.size(); ++i) {
    used_inputs.insert(i);
  }
  std::tie(input_descs, input_col_descs) = get_input_desc_impl(filter, rte_idx, used_inputs);
  const auto qual = translate_scalar_rex(filter->getCondition(), rte_idx, cat_, in_metainfo);
  const auto target_exprs_owned = synthesize_inputs(filter, in_metainfo, rte_idx);
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  Executor::RelAlgExecutionUnit rel_alg_exe_unit{
      input_descs, input_col_descs, {}, {qual}, {}, {nullptr}, target_exprs, {}, 0};
  return executeWorkUnit(rel_alg_exe_unit, input_descs, in_metainfo, false, co);
}

ExecutionResult RelAlgExecutor::executeWorkUnit(const Executor::RelAlgExecutionUnit& rel_alg_exe_unit,
                                                const std::vector<InputDescriptor>& input_descs,
                                                const std::vector<TargetMetaInfo>& targets_meta,
                                                const bool is_agg,
                                                const CompilationOptions& co) {
  size_t max_groups_buffer_entry_guess{2048};
  int32_t error_code{0};
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;
  return {executor_->executeWorkUnit(&error_code,
                                     max_groups_buffer_entry_guess,
                                     is_agg,
                                     get_table_infos(input_descs, cat_, temporary_tables_),
                                     rel_alg_exe_unit,
                                     co,
                                     {false, true, false, false},
                                     cat_,
                                     executor_->row_set_mem_owner_,
                                     nullptr),
          targets_meta};
}

#endif  // HAVE_CALCITE
