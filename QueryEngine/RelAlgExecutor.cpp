#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"
#include "RelAlgTranslator.h"

#include "InputMetadata.h"
#include "RexVisitor.h"
#include "ScalarExprVisitor.h"

ExecutionResult RelAlgExecutor::executeRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo) {
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  decltype(temporary_tables_)().swap(temporary_tables_);
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;
  time(&now_);
  CHECK(!exec_descs.empty());
  const auto exec_desc_count = eo.just_explain ? size_t(1) : exec_descs.size();
  for (size_t i = 0; i < exec_desc_count; ++i) {
    auto& exec_desc = exec_descs[i];
    const auto body = exec_desc.getBody();
    const auto compound = dynamic_cast<const RelCompound*>(body);
    if (compound) {
      exec_desc.setResult(executeCompound(compound, co, eo));
      addTemporaryTable(-compound->getId(), &exec_desc.getResult().getRows());
      continue;
    }
    const auto project = dynamic_cast<const RelProject*>(body);
    if (project) {
      exec_desc.setResult(executeProject(project, co, eo));
      addTemporaryTable(-project->getId(), &exec_desc.getResult().getRows());
      continue;
    }
    const auto filter = dynamic_cast<const RelFilter*>(body);
    if (filter) {
      exec_desc.setResult(executeFilter(filter, co, eo));
      addTemporaryTable(-filter->getId(), &exec_desc.getResult().getRows());
      continue;
    }
    const auto sort = dynamic_cast<const RelSort*>(body);
    if (sort) {
      exec_desc.setResult(executeSort(sort, co, eo));
      addTemporaryTable(-sort->getId(), &exec_desc.getResult().getRows());
      continue;
    }
    CHECK(false);
  }
  return exec_descs[exec_desc_count - 1].getResult();
}

namespace {

class RexUsedInputsVisitor : public RexVisitor<std::unordered_set<const RexInput*>> {
 public:
  std::unordered_set<const RexInput*> visitInput(const RexInput* rex_input) const override { return {rex_input}; }

 protected:
  std::unordered_set<const RexInput*> aggregateResult(
      const std::unordered_set<const RexInput*>& aggregate,
      const std::unordered_set<const RexInput*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

std::unordered_set<const RexInput*> get_used_inputs(const RelCompound* compound) {
  RexUsedInputsVisitor visitor;
  const auto filter_expr = compound->getFilterExpr();
  std::unordered_set<const RexInput*> used_inputs =
      filter_expr ? visitor.visit(filter_expr) : std::unordered_set<const RexInput*>{};
  const auto sources_size = compound->getScalarSourcesSize();
  for (size_t i = 0; i < sources_size; ++i) {
    const auto source_inputs = visitor.visit(compound->getScalarSource(i));
    used_inputs.insert(source_inputs.begin(), source_inputs.end());
  }
  return used_inputs;
}

std::unordered_set<const RexInput*> get_used_inputs(const RelProject* project) {
  RexUsedInputsVisitor visitor;
  std::unordered_set<const RexInput*> used_inputs;
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

const RelAlgNode* get_data_sink(const RelAlgNode* ra_node) {
  CHECK_EQ(size_t(1), ra_node->inputCount());
  const auto join_input = dynamic_cast<const RelJoin*>(ra_node->getInput(0));
  // If the input node is a join, the data is sourced from it instead of the initial node.
  const auto data_sink_node =
      join_input ? static_cast<const RelAlgNode*>(join_input) : static_cast<const RelAlgNode*>(ra_node);
  CHECK(1 <= data_sink_node->inputCount() && data_sink_node->inputCount() <= 2);
  return data_sink_node;
}

std::unordered_map<const RelAlgNode*, int> get_input_nest_levels(const RelAlgNode* ra_node) {
  const auto data_sink_node = get_data_sink(ra_node);
  std::unordered_map<const RelAlgNode*, int> input_to_nest_level;
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto input_ra = data_sink_node->getInput(nest_level);
    const auto it_ok = input_to_nest_level.emplace(input_ra, nest_level);
    CHECK(it_ok.second);
  }
  return input_to_nest_level;
}

template <class RA>
std::pair<std::vector<InputDescriptor>, std::list<InputColDescriptor>> get_input_desc_impl(
    const RA* ra_node,
    const std::unordered_set<const RexInput*>& used_inputs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::vector<InputDescriptor> input_descs;
  const auto data_sink_node = get_data_sink(ra_node);
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto input_ra = data_sink_node->getInput(nest_level);
    const int table_id = table_id_from_ra(input_ra);
    input_descs.emplace_back(table_id, nest_level);
  }
  std::unordered_set<InputColDescriptor> input_col_descs_unique;
  for (const auto used_input : used_inputs) {
    const auto input_ra = used_input->getSourceNode();
    const auto scan_ra = dynamic_cast<const RelScan*>(input_ra);
    const int table_id = table_id_from_ra(input_ra);
    const auto it = input_to_nest_level.find(input_ra);
    CHECK(it != input_to_nest_level.end());
    // Physical columns from a scan node are numbered from 1 in our system.
    input_col_descs_unique.emplace(scan_ra ? used_input->getIndex() + 1 : used_input->getIndex(), table_id, it->second);
  }
  return {input_descs, std::list<InputColDescriptor>(input_col_descs_unique.begin(), input_col_descs_unique.end())};
}

template <class RA>
std::pair<std::vector<InputDescriptor>, std::list<InputColDescriptor>> get_input_desc(
    const RA* ra_node,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  const auto used_inputs = get_used_inputs(ra_node);
  return get_input_desc_impl(ra_node, used_inputs, input_to_nest_level);
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

std::shared_ptr<Analyzer::Expr> set_transient_dict(const std::shared_ptr<Analyzer::Expr> expr) {
  const auto& ti = expr->get_type_info();
  if (!ti.is_string() || ti.get_compression() != kENCODING_NONE) {
    return expr;
  }
  auto transient_dict_ti = ti;
  transient_dict_ti.set_compression(kENCODING_DICT);
  transient_dict_ti.set_comp_param(TRANSIENT_DICT_ID);
  transient_dict_ti.set_fixed_size();
  return expr->add_cast(transient_dict_ti);
}

template <class RA>
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(const RA* ra_node,
                                                                      const RelAlgTranslator& translator) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < get_scalar_sources_size(ra_node); ++i) {
    scalar_sources.push_back(translator.translateScalarRex(scalar_at(i, ra_node)));
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
  for (size_t group_idx = 0; group_idx < compound->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}

struct QualsConjunctiveForm {
  const std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> quals;
};

QualsConjunctiveForm qual_to_conjunctive_form(const std::shared_ptr<Analyzer::Expr> qual_expr) {
  CHECK(qual_expr);
  const auto bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(qual_expr);
  if (!bin_oper) {
    return {{}, {qual_expr}};
  }
  if (bin_oper->get_optype() == kAND) {
    const auto lhs_cf = qual_to_conjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_cf = qual_to_conjunctive_form(bin_oper->get_own_right_operand());
    auto simple_quals = lhs_cf.simple_quals;
    simple_quals.insert(simple_quals.end(), rhs_cf.simple_quals.begin(), rhs_cf.simple_quals.end());
    auto quals = lhs_cf.quals;
    quals.insert(quals.end(), rhs_cf.quals.begin(), rhs_cf.quals.end());
    return {simple_quals, quals};
  }
  int rte_idx{0};
  const auto simple_qual = bin_oper->normalize_simple_predicate(rte_idx);
  return simple_qual ? QualsConjunctiveForm{{simple_qual}, {}} : QualsConjunctiveForm{{}, {qual_expr}};
}

QualsConjunctiveForm translate_quals(const RelCompound* compound, const RelAlgTranslator& translator) {
  const auto filter_rex = compound->getFilterExpr();
  const auto filter_expr = filter_rex ? translator.translateScalarRex(filter_rex) : nullptr;
  return filter_expr ? qual_to_conjunctive_form(filter_expr) : QualsConjunctiveForm{};
}

std::vector<Analyzer::Expr*> translate_targets(std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
                                               const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
                                               const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                                               const RelCompound* compound,
                                               const RelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  for (size_t i = 0; i < compound->size(); ++i) {
    const auto target_rex = compound->getTargetExpr(i);
    const auto target_rex_agg = dynamic_cast<const RexAgg*>(target_rex);
    std::shared_ptr<Analyzer::Expr> target_expr;
    if (target_rex_agg) {
      target_expr = RelAlgTranslator::translateAggregateRex(target_rex_agg, scalar_sources);
    } else {
      const auto target_rex_scalar = dynamic_cast<const RexScalar*>(target_rex);
      const auto target_rex_ref = dynamic_cast<const RexRef*>(target_rex_scalar);
      if (target_rex_ref) {
        const auto ref_idx = target_rex_ref->getIndex();
        CHECK_GE(ref_idx, size_t(1));
        CHECK_LE(ref_idx, groupby_exprs.size());
        const auto groupby_expr = *std::next(groupby_exprs.begin(), ref_idx - 1);
        target_expr = var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, ref_idx);
      } else {
        target_expr = translator.translateScalarRex(target_rex_scalar);
      }
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
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo) {
  const auto work_unit = createCompoundWorkUnit(compound, {});
  return executeWorkUnit(work_unit, compound->getOutputMetainfo(), compound->isAggregate(), co, eo);
}

ExecutionResult RelAlgExecutor::executeProject(const RelProject* project,
                                               const CompilationOptions& co,
                                               const ExecutionOptions& eo) {
  const auto work_unit = createProjectWorkUnit(project, {});
  return executeWorkUnit(work_unit, project->getOutputMetainfo(), false, co, eo);
}

ExecutionResult RelAlgExecutor::executeFilter(const RelFilter* filter,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo) {
  const auto work_unit = createFilterWorkUnit(filter, {});
  return executeWorkUnit(work_unit, filter->getOutputMetainfo(), false, co, eo);
}

namespace {

// TODO(alex): Once we're fully migrated to the relational algebra model, change
// the executor interface to use the collation directly and remove this conversion.
std::list<Analyzer::OrderEntry> get_order_entries(const RelSort* sort) {
  std::list<Analyzer::OrderEntry> result;
  for (size_t i = 0; i < sort->collationCount(); ++i) {
    const auto sort_field = sort->getCollation(i);
    result.emplace_back(sort_field.getField() + 1,
                        sort_field.getSortDir() == SortDirection::Descending,
                        sort_field.getNullsPosition() == NullSortedPosition::First);
  }
  return result;
}

size_t get_scan_limit(const RelAlgNode* ra, const size_t limit) {
  const auto compound = dynamic_cast<const RelCompound*>(ra);
  return (compound && compound->isAggregate()) ? 0 : limit;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeSort(const RelSort* sort,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo) {
  CHECK_EQ(size_t(1), sort->inputCount());
  const auto source = sort->getInput(0);
  CHECK(!dynamic_cast<const RelSort*>(source));
  const auto compound = dynamic_cast<const RelCompound*>(source);
  const auto source_work_unit = createSortInputWorkUnit(sort);
  auto source_result = executeWorkUnit(
      source_work_unit, source->getOutputMetainfo(), compound ? compound->isAggregate() : false, co, eo);
  auto rows_to_sort = source_result.getRows();
  if (eo.just_explain) {
    return {rows_to_sort, {}};
  }
  const size_t limit = sort->getLimit();
  const size_t offset = sort->getOffset();
  if (sort->collationCount() != 0) {
    rows_to_sort.sort(source_work_unit.exe_unit.order_entries, false, limit + offset);
  }
  if (limit || offset) {
    rows_to_sort.dropFirstN(offset);
    if (limit) {
      rows_to_sort.keepFirstN(limit);
    }
  }
  return {rows_to_sort, source_result.getTargetsMeta()};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createSortInputWorkUnit(const RelSort* sort) {
  const auto source = sort->getInput(0);
  auto source_work_unit = createWorkUnit(source, get_order_entries(sort));
  const size_t limit = sort->getLimit();
  const size_t offset = sort->getOffset();
  const size_t scan_limit = sort->collationCount() ? 0 : get_scan_limit(source, limit);
  const size_t scan_total_limit = scan_limit ? get_scan_limit(source, scan_limit + offset) : 0;
  size_t max_groups_buffer_entry_guess{scan_total_limit ? scan_total_limit : max_groups_buffer_entry_default_guess};
  const auto& source_exe_unit = source_work_unit.exe_unit;
  return {{source_exe_unit.input_descs,
           source_exe_unit.input_col_descs,
           source_exe_unit.simple_quals,
           source_exe_unit.quals,
           source_exe_unit.join_quals,
           source_exe_unit.groupby_exprs,
           source_exe_unit.target_exprs,
           source_exe_unit.order_entries,
           scan_total_limit},
          max_groups_buffer_entry_guess,
          std::move(source_work_unit.query_rewriter)};
}

ExecutionResult RelAlgExecutor::executeWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                                const std::vector<TargetMetaInfo>& targets_meta,
                                                const bool is_agg,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo) {
  int32_t error_code{0};
  size_t max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  ExecutionResult result = {
      executor_->executeWorkUnit(&error_code,
                                 max_groups_buffer_entry_guess,
                                 is_agg,
                                 get_table_infos(work_unit.exe_unit.input_descs, cat_, temporary_tables_),
                                 work_unit.exe_unit,
                                 co,
                                 eo,
                                 cat_,
                                 executor_->row_set_mem_owner_,
                                 nullptr),
      targets_meta};
  if (!error_code) {
    return result;
  }
  if (error_code == Executor::ERR_DIV_BY_ZERO) {
    throw std::runtime_error("Division by zero");
  }
  if (error_code == Executor::ERR_UNSUPPORTED_SELF_JOIN) {
    throw std::runtime_error("Self joins not supported yet");
  }
  if (error_code == Executor::ERR_OUT_OF_CPU_MEM) {
    throw std::runtime_error("Not enough host memory to execute the query");
  }
  return handleRetry(error_code, {work_unit.exe_unit, max_groups_buffer_entry_guess}, targets_meta, is_agg, co, eo);
}

ExecutionResult RelAlgExecutor::handleRetry(const int32_t error_code_in,
                                            const RelAlgExecutor::WorkUnit& work_unit,
                                            const std::vector<TargetMetaInfo>& targets_meta,
                                            const bool is_agg,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo) {
  auto error_code = error_code_in;
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  ExecutionOptions eo_no_multifrag{eo.output_columnar_hint, false, false, eo.allow_loop_joins};
  ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, co.device_type_), {}};
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM) {
    result = {executor_->executeWorkUnit(&error_code,
                                         max_groups_buffer_entry_guess,
                                         is_agg,
                                         get_table_infos(work_unit.exe_unit.input_descs, cat_, temporary_tables_),
                                         work_unit.exe_unit,
                                         co,
                                         eo_no_multifrag,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         nullptr),
              targets_meta};
  }
  CompilationOptions co_cpu{ExecutorDeviceType::CPU, co.hoist_literals_, co.opt_level_};
  if (error_code) {
    max_groups_buffer_entry_guess = 0;
    while (true) {
      result = {executor_->executeWorkUnit(&error_code,
                                           max_groups_buffer_entry_guess,
                                           is_agg,
                                           get_table_infos(work_unit.exe_unit.input_descs, cat_, temporary_tables_),
                                           work_unit.exe_unit,
                                           co_cpu,
                                           eo_no_multifrag,
                                           cat_,
                                           executor_->row_set_mem_owner_,
                                           nullptr),
                targets_meta};
    }
    if (!error_code) {
      return result;
    }
    // Even the conservative guess failed; it should only happen when we group
    // by a huge cardinality array. Maybe we should throw an exception instead?
    // Such a heavy query is entirely capable of exhausting all the host memory.
    CHECK(max_groups_buffer_entry_guess);
    max_groups_buffer_entry_guess *= 2;
  }
  return result;
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createWorkUnit(const RelAlgNode* node,
                                                        const std::list<Analyzer::OrderEntry>& order_entries) {
  const auto compound = dynamic_cast<const RelCompound*>(node);
  if (compound) {
    return createCompoundWorkUnit(compound, order_entries);
  }
  const auto project = dynamic_cast<const RelProject*>(node);
  if (project) {
    return createProjectWorkUnit(project, order_entries);
  }
  const auto filter = dynamic_cast<const RelFilter*>(node);
  CHECK(filter);
  return createFilterWorkUnit(filter, order_entries);
}

namespace {

class UsedTablesVisitor : public ScalarExprVisitor<std::unordered_set<int>> {
 protected:
  virtual std::unordered_set<int> visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return {column->get_table_id()};
  }

  virtual std::unordered_set<int> aggregateResult(const std::unordered_set<int>& aggregate,
                                                  const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

struct SeparatedQuals {
  const std::list<std::shared_ptr<Analyzer::Expr>> regular_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
};

SeparatedQuals separate_join_quals(const std::list<std::shared_ptr<Analyzer::Expr>>& all_quals) {
  std::list<std::shared_ptr<Analyzer::Expr>> regular_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
  UsedTablesVisitor qual_visitor;
  for (auto qual_candidate : all_quals) {
    const auto used_table_ids = qual_visitor.visit(qual_candidate.get());
    if (used_table_ids.size() > 1) {
      CHECK_EQ(size_t(2), used_table_ids.size());
      join_quals.push_back(qual_candidate);
    } else {
      regular_quals.push_back(qual_candidate);
    }
  }
  return {regular_quals, join_quals};
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createCompoundWorkUnit(const RelCompound* compound,
                                                                const std::list<Analyzer::OrderEntry>& order_entries) {
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  const auto input_to_nest_level = get_input_nest_levels(compound);
  std::tie(input_descs, input_col_descs) = get_input_desc(compound, input_to_nest_level);
  RelAlgTranslator translator(cat_, input_to_nest_level, now_);
  const auto scalar_sources = translate_scalar_sources(compound, translator);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  const auto separated_quals = separate_join_quals(quals_cf.quals);
  const auto simple_separated_quals = separate_join_quals(quals_cf.simple_quals);
  CHECK(simple_separated_quals.join_quals.empty());
  const auto target_exprs = translate_targets(target_exprs_owned_, scalar_sources, groupby_exprs, compound, translator);
  CHECK_EQ(compound->size(), target_exprs.size());
  const Executor::RelAlgExecutionUnit exe_unit = {input_descs,
                                                  input_col_descs,
                                                  quals_cf.simple_quals,
                                                  separated_quals.regular_quals,
                                                  separated_quals.join_quals,
                                                  groupby_exprs,
                                                  target_exprs,
                                                  order_entries,
                                                  0};
  const auto query_infos = get_table_infos(exe_unit.input_descs, cat_, temporary_tables_);
  QueryRewriter* query_rewriter = new QueryRewriter(exe_unit, query_infos, executor_, nullptr);
  const auto rewritten_exe_unit = query_rewriter->rewrite();
  const auto targets_meta = get_targets_meta(compound, rewritten_exe_unit.target_exprs);
  compound->setOutputMetainfo(targets_meta);
  return {rewritten_exe_unit, max_groups_buffer_entry_default_guess, std::unique_ptr<QueryRewriter>(query_rewriter)};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createProjectWorkUnit(const RelProject* project,
                                                               const std::list<Analyzer::OrderEntry>& order_entries) {
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  const auto input_to_nest_level = get_input_nest_levels(project);
  std::tie(input_descs, input_col_descs) = get_input_desc(project, input_to_nest_level);
  RelAlgTranslator translator(cat_, input_to_nest_level, now_);
  const auto target_exprs_owned = translate_scalar_sources(project, translator);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  const auto targets_meta = get_targets_meta(project, target_exprs);
  project->setOutputMetainfo(targets_meta);
  return {{input_descs, input_col_descs, {}, {}, {}, {nullptr}, target_exprs, order_entries, 0},
          max_groups_buffer_entry_default_guess,
          nullptr};
}

namespace {

std::vector<std::shared_ptr<Analyzer::Expr>> synthesize_inputs(
    const RelAlgNode* ra_node,
    const std::vector<TargetMetaInfo>& in_metainfo,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  CHECK_EQ(size_t(1), ra_node->inputCount());
  const auto input = ra_node->getInput(0);
  const auto it_rte_idx = input_to_nest_level.find(input);
  CHECK(it_rte_idx != input_to_nest_level.end());
  const int rte_idx = it_rte_idx->second;
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

RelAlgExecutor::WorkUnit RelAlgExecutor::createFilterWorkUnit(const RelFilter* filter,
                                                              const std::list<Analyzer::OrderEntry>& order_entries) {
  CHECK_EQ(size_t(1), filter->inputCount());
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::unique_ptr<RexInput>> used_inputs_owned;
  const auto source = filter->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  for (size_t i = 0; i < in_metainfo.size(); ++i) {
    auto synthesized_used_input = new RexInput(source, i);
    used_inputs_owned.emplace_back(synthesized_used_input);
    used_inputs.insert(synthesized_used_input);
  }
  const auto input_to_nest_level = get_input_nest_levels(filter);
  std::tie(input_descs, input_col_descs) = get_input_desc_impl(filter, used_inputs, input_to_nest_level);
  RelAlgTranslator translator(cat_, input_to_nest_level, now_);
  const auto qual = translator.translateScalarRex(filter->getCondition());
  const auto target_exprs_owned = synthesize_inputs(filter, in_metainfo, input_to_nest_level);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  filter->setOutputMetainfo(in_metainfo);
  return {{input_descs, input_col_descs, {}, {qual}, {}, {nullptr}, target_exprs, order_entries, 0},
          max_groups_buffer_entry_default_guess,
          nullptr};
}

#endif  // HAVE_CALCITE
