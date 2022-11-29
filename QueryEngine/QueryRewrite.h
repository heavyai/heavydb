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

  RelAlgExecutionUnit rewriteCountIfAggRemoveNonNull(
      const RelAlgExecutionUnit& ra_exe_unit_in) const;

 private:
  RelAlgExecutionUnit rewriteOverlapsJoin(
      const RelAlgExecutionUnit& ra_exe_unit_in) const;

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
