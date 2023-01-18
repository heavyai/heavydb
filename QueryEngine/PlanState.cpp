/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PlanState.h"
#include "Execute.h"

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

  if (!shared::is_unordered_set_intersection_empty(columns_to_fetch_,
                                                   columns_to_not_fetch_)) {
    if (columns_to_fetch_.count(column_key) && columns_to_not_fetch_.count(column_key) &&
        hasGeometryTargetExpr()) {
      // when a query having a projection expression of the non-point geometry type,
      // we must avoid throwing a false `CompilationRetryNoLazyFetch` exception
      // because the non-point geometry projection expression must be lazy fetched
      // this can be achieved by making an expression as non-lazy fetched column iff
      // it is included as both lazy and non-lazy fetched column simultaneously
      // (we throw the exception for that case; see the previous code of this function)
      VLOG(2) << "Set a column " << do_not_fetch_column->toString()
              << " as non-lazy fetching column";
      columns_to_not_fetch_.erase(column_key);
    } else {
      throw CompilationRetryNoLazyFetch();
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
      columns_to_fetch_.insert(global_col_key);
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

bool PlanState::hasGeometryTargetExpr() const {
  return std::any_of(
      target_exprs_.begin(), target_exprs_.end(), [=](Analyzer::Expr const* expr) {
        return expr->get_type_info().is_geometry();
      });
}
