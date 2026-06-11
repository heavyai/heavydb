/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <list>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "Analyzer/Analyzer.h"
#include "Fragmenter/Fragmenter.h"
#include "QueryEngine/Execute.h"

class QueryRewriter {
 public:
  QueryRewriter(const std::vector<InputTableInfo>& query_infos, Executor* executor)
      : query_infos_(query_infos), executor_(executor) {}
  RelAlgExecutionUnit rewrite(const RelAlgExecutionUnit& ra_exe_unit_in) const;

  RelAlgExecutionUnit rewriteColumnarUpdate(
      const RelAlgExecutionUnit& ra_exe_unit_in,
      std::shared_ptr<Analyzer::ColumnVar> column_to_update) const;

  RelAlgExecutionUnit rewriteColumnarDelete(
      const RelAlgExecutionUnit& ra_exe_unit_in,
      std::shared_ptr<Analyzer::ColumnVar> delete_column) const;

  RelAlgExecutionUnit rewriteAggregateOnGroupByColumn(
      const RelAlgExecutionUnit& ra_exe_unit_in) const;

 private:
  RelAlgExecutionUnit rewriteConstrainedByIn(
      const RelAlgExecutionUnit& ra_exe_unit_in) const;

  RelAlgExecutionUnit rewriteConstrainedByInImpl(
      const RelAlgExecutionUnit& ra_exe_unit_in,
      const std::shared_ptr<Analyzer::CaseExpr>,
      const Analyzer::InValues*) const;

  static std::shared_ptr<Analyzer::CaseExpr> generateCaseForDomainValues(
      const Analyzer::InValues*);

  std::pair<bool, std::set<size_t>> is_all_groupby_exprs_are_col_var(
      const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs) const;

  std::shared_ptr<Analyzer::CaseExpr> generateCaseExprForCountDistinctOnGroupByCol(
      std::shared_ptr<Analyzer::Expr> expr) const;

  const std::vector<InputTableInfo>& query_infos_;
  Executor* executor_;
  mutable std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;
};
