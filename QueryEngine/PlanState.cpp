/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PlanState.h"

#include "Execute.h"
#include "Shared/misc.h"
#include "Visitors/CommonVisitors.h"

extern bool g_enable_result_reduction_pipeline;

namespace {

class ComparisonColumnsVisitor
    : public ScalarExprVisitor<std::set<const Analyzer::ColumnVar*>> {
 protected:
  std::set<const Analyzer::ColumnVar*> visitColumnVar(
      const Analyzer::ColumnVar*) const override {
    return {};
  }

  std::set<const Analyzer::ColumnVar*> visitColumnVarTuple(
      const Analyzer::ExpressionTuple* expr_tuple) const override {
    AllColumnVarsVisitor visitor;
    std::set<const Analyzer::ColumnVar*> result;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      const auto component_columns = visitor.visit(expr_component.get());
      result.insert(component_columns.begin(), component_columns.end());
    }
    return result;
  }

  std::set<const Analyzer::ColumnVar*> visitBinOper(
      const Analyzer::BinOper* bin_oper) const override {
    if (IS_COMPARISON(bin_oper->get_optype())) {
      AllColumnVarsVisitor visitor;
      auto result = visitor.visit(bin_oper->get_left_operand());
      const auto rhs_columns = visitor.visit(bin_oper->get_right_operand());
      result.insert(rhs_columns.begin(), rhs_columns.end());
      return result;
    }
    auto result = visit(bin_oper->get_left_operand());
    const auto rhs_columns = visit(bin_oper->get_right_operand());
    result.insert(rhs_columns.begin(), rhs_columns.end());
    return result;
  }

  std::set<const Analyzer::ColumnVar*> aggregateResult(
      const std::set<const Analyzer::ColumnVar*>& aggregate,
      const std::set<const Analyzer::ColumnVar*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

}  // namespace

bool PlanState::isLazyFetchColumn(const Analyzer::Expr* target_expr) const {
  if (!allow_lazy_fetch_) {
    return false;
  }
  const auto do_not_fetch_column = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
  if (!do_not_fetch_column || dynamic_cast<const Analyzer::Var*>(do_not_fetch_column)) {
    return false;
  }
  const auto& column_key = do_not_fetch_column->getColumnKey();
  if (column_key.table_id > 0) {
    const auto cd = get_column_descriptor(column_key);
    if (cd->isVirtualCol) {
      return false;
    }
  }
  return columns_to_fetch_.find(column_key) == columns_to_fetch_.end();
}

void PlanState::allocateLocalColumnIds(
    const std::list<std::shared_ptr<const InputColDescriptor>>& global_col_ids) {
  for (const auto& col_id : global_col_ids) {
    CHECK(col_id);
    const auto local_col_id = global_to_local_col_ids_.size();
    const auto it_ok =
        global_to_local_col_ids_.insert(std::make_pair(*col_id, local_col_id));
    // enforce uniqueness of the column ids in the scan plan
    CHECK(it_ok.second);
  }
}

int PlanState::getLocalColumnId(const Analyzer::ColumnVar* col_var,
                                const bool fetch_column) {
  // Previously, we consider `rte_idx` of `col_var` w/ its column key together
  // to specify columns in the `global_to_local_col_ids_`.
  // But there is a case when the same col has multiple 'rte_idx's
  // For instance, the same geometry col is used not only as input col of the geo join op,
  // but also included as input col of filter predicate
  // In such a case, the same geometry col has two rte_idxs (the one defined by the filter
  // predicate and the other determined by the geo join operator)
  // The previous logic cannot cover this case b/c it allows only one `rte_idx` per col
  // But it is safe to share `rte_idx` of among all use cases of the same col
  CHECK(col_var);
  const auto& global_col_key = col_var->getColumnKey();
  InputColDescriptor scan_col_desc(global_col_key.column_id,
                                   global_col_key.table_id,
                                   global_col_key.db_id,
                                   col_var->get_rte_idx());
  std::optional<int> col_id{std::nullopt};
  // let's try to find col_id w/ considering `rte_idx`
  const auto it = global_to_local_col_ids_.find(scan_col_desc);
  if (it != global_to_local_col_ids_.end()) {
    // we have a valid col_id
    col_id = it->second;
  } else {
    // otherwise, let's try to find col_id for the same col
    // (but have different 'rte_idx') to share it w/ `col_var`
    for (auto const& kv : global_to_local_col_ids_) {
      if (kv.first.getColumnKey() == global_col_key) {
        col_id = kv.second;
        break;
      }
    }
  }
  if (col_id && *col_id >= 0) {
    if (fetch_column) {
      addColumnToFetch(global_col_key);
    }
    return *col_id;
  }
  CHECK(false) << "Expected to find " << global_col_key;
  return {};
}

void PlanState::addNonHashtableQualForLeftJoin(size_t idx,
                                               std::shared_ptr<Analyzer::Expr> expr) {
  auto it = left_join_non_hashtable_quals_.find(idx);
  if (it == left_join_non_hashtable_quals_.end()) {
    std::vector<std::shared_ptr<Analyzer::Expr>> expr_vec{expr};
    left_join_non_hashtable_quals_.emplace(idx, std::move(expr_vec));
  } else {
    it->second.emplace_back(expr);
  }
}

const std::unordered_set<shared::ColumnKey>& PlanState::getColumnsToFetch() const {
  return columns_to_fetch_;
}

const std::unordered_set<shared::ColumnKey>& PlanState::getColumnsToNotFetch() const {
  return columns_to_not_fetch_;
}

bool PlanState::isColumnToFetch(const shared::ColumnKey& column_key) const {
  return shared::contains(columns_to_fetch_, column_key);
}

bool PlanState::isColumnToNotFetch(const shared::ColumnKey& column_key) const {
  return shared::contains(columns_to_not_fetch_, column_key);
}

bool PlanState::isColumnToFetchHostMapped(const shared::ColumnKey& column_key) const {
  return shared::contains(host_mapped_columns_to_fetch_, column_key);
}

bool PlanState::isColumnToFetchSelectedDense(const shared::ColumnKey& column_key) const {
  return shared::contains(selected_dense_columns_to_fetch_, column_key);
}

bool PlanState::hasSelectedDenseColumnsToFetch() const {
  return !selected_dense_columns_to_fetch_.empty();
}

bool PlanState::isColumnToFetchSegmented(const InputColDescriptor& column_desc) const {
  return shared::contains(segmented_columns_to_fetch_, column_desc);
}

bool PlanState::canUseSegmentedColumnFetch(const InputColDescriptor& column_desc,
                                           const SQLTypeInfo& type_info,
                                           const ExecutorDeviceType device_type) const {
  return g_enable_result_reduction_pipeline && device_type == ExecutorDeviceType::GPU &&
         column_desc.getScanDesc().getSourceType() == InputSourceType::RESULT &&
         !type_info.is_array() && !type_info.is_geometry() && !type_info.is_varlen() &&
         !type_info.usesFlatBuffer() && type_info.get_size() > 0;
}

// todo (yoonmin): determine non-lazy fetch compilation in a high-level; to avoid
// recompilation which may incur noticeable overhead
void PlanState::addColumnToFetch(const shared::ColumnKey& column_key,
                                 bool unmark_lazy_fetch) {
  if (unmark_lazy_fetch) {
    // This column was previously marked for lazy fetch but we now intentionally
    // mark it as non-lazy fetch column
    auto it = columns_to_not_fetch_.find(column_key);
    if (it != columns_to_not_fetch_.end()) {
      columns_to_not_fetch_.erase(it);
    }
  } else {
    if (columns_to_not_fetch_.count(column_key)) {
      // this case represents a query w/ fetching expressions with the order of
      // cv and an expression on it, i.e., SELECT col1, ST_NPoints(col1) FROM ...
      // since we first touch the cv itself and we forcely mark it as lazy-fetch
      // but to evaluate the expression, we must fetch the same cv;
      // so let's fetch the cv to allow evaluating the expression using the same cv
      throw CompilationRetryNoLazyFetch();
    }
  }
  CHECK(!isColumnToNotFetch(column_key)) << column_key;
  columns_to_fetch_.emplace(column_key);
}

void PlanState::addColumnToNotFetch(const shared::ColumnKey& column_key) {
  CHECK(!isColumnToFetch(column_key)) << column_key;
  columns_to_not_fetch_.emplace(column_key);
}

void PlanState::addColumnToFetchHostMapped(const shared::ColumnKey& column_key) {
  host_mapped_columns_to_fetch_.emplace(column_key);
}

void PlanState::addColumnToFetchSelectedDense(const shared::ColumnKey& column_key) {
  selected_dense_columns_to_fetch_.emplace(column_key);
}

void PlanState::addColumnToFetchSegmented(const InputColDescriptor& column_desc) {
  segmented_columns_to_fetch_.emplace(column_desc);
}

void PlanState::addColumnToFetchSegmented(const InputColDescriptor& column_desc,
                                          const bool unmark_lazy_fetch) {
  addColumnToFetch(column_desc.getColumnKey(), unmark_lazy_fetch);
  addColumnToFetchSegmented(column_desc);
}

bool PlanState::hasExpressionNeedsLazyFetch(
    const std::vector<TargetExprCodegen>& target_exprs_to_codegen) const {
  return std::any_of(target_exprs_to_codegen.begin(),
                     target_exprs_to_codegen.end(),
                     [](const TargetExprCodegen& target_expr) {
                       return target_expr.target_info.sql_type.usesFlatBuffer();
                     });
}

void PlanState::registerNonLazyFetchExpression(
    const std::vector<TargetExprCodegen>& target_exprs_to_codegen) {
  auto const needs_lazy_fetch = hasExpressionNeedsLazyFetch(target_exprs_to_codegen);
  ComparisonColumnsVisitor comparison_columns_visitor;
  for (const auto& expr : target_exprs_to_codegen) {
    if (needs_lazy_fetch && !expr.target_info.sql_type.usesFlatBuffer()) {
      if (auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr.target_expr)) {
        // force non-lazy fetch on all other columns that don't support flatbuffer
        addColumnToFetch(col_var->getColumnKey(), /*unmark_lazy_fetch=*/true);
      }
    }
    const auto comparison_columns = comparison_columns_visitor.visit(expr.target_expr);
    for (const auto column : comparison_columns) {
      if (isLazyFetchColumn(column)) {
        addColumnToFetch(column->getColumnKey(), /*unmark_lazy_fetch=*/true);
      }
    }
  }
}
