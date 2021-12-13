/*
 * Copyright 2019 OmniSci, Inc.
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
  if (do_not_fetch_column->get_table_id() > 0) {
    auto cd = get_column_descriptor(do_not_fetch_column->get_column_id(),
                                    do_not_fetch_column->get_table_id(),
                                    *executor_->getCatalog());
    if (cd->isVirtualCol) {
      return false;
    }
  }
  InputColDescriptorSet intersect;
  std::set_intersection(columns_to_fetch_.begin(),
                        columns_to_fetch_.end(),
                        columns_to_not_fetch_.begin(),
                        columns_to_not_fetch_.end(),
                        std::inserter(intersect, intersect.begin()),
                        CompareInputColDescId());
  if (!intersect.empty()) {
    throw CompilationRetryNoLazyFetch();
  }
  return columns_to_fetch_.find(column_var_to_descriptor(do_not_fetch_column)) ==
         columns_to_fetch_.end();
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
  CHECK(col_var);
  InputColDescriptor scan_col_desc(col_var->get_column_info(), col_var->get_rte_idx());
  const auto it = global_to_local_col_ids_.find(scan_col_desc);
  CHECK(it != global_to_local_col_ids_.end()) << "Expected to find " << scan_col_desc;
  if (fetch_column) {
    columns_to_fetch_.insert(column_var_to_descriptor(col_var));
  }
  return it->second;
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
