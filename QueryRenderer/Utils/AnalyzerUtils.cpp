/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/AnalyzerUtils.h"

#include <set>

#include "Analyzer/Analyzer.h"

namespace {
void collect_column_var(const Analyzer::Expr* expr,
                        std::set<const Analyzer::ColumnVar*,
                                 bool (*)(const Analyzer::ColumnVar*,
                                          const Analyzer::ColumnVar*)>& colvar_set,
                        bool include_agg) {
  auto case_var = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_var) {
    const auto expr_pair_list = case_var->get_expr_pair_list();
    const auto else_expr = case_var->get_else_expr();
    // only checking the THEN expr of case statements as that's
    // the only contributor to the output of the CASE. We're going
    // to ignore the expr in the WHEN
    // Needed to add our own specialization for CASE expr to
    // handle this appropriately.
    for (auto p : expr_pair_list) {
      // collecting column vars from the THEN expr only,
      // ignoring the WHEN expr
      p.second->collect_column_var(colvar_set, include_agg);
    }
    if (else_expr != nullptr) {
      else_expr->collect_column_var(colvar_set, include_agg);
    }
  } else {
    expr->collect_column_var(colvar_set, include_agg);
  }
}
}  // namespace

namespace QueryRenderer {

std::pair<int, int> get_table_id_col_id_from_target_expr(const Analyzer::Expr* expr) {
  int tableid = -1, colid = -1;
  auto fn_pt = Analyzer::ColumnVar::colvar_comp;
  std::set<const Analyzer::ColumnVar*,
           bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
      colvar_set(fn_pt);
  collect_column_var(expr, colvar_set, true);
  if (colvar_set.size() == 1) {
    auto itr = colvar_set.begin();
    const auto& column_key = (*itr)->getColumnKey();
    tableid = column_key.table_id;
    colid = column_key.column_id;
  }
  return std::make_pair(tableid, colid);
}

}  // namespace QueryRenderer
