/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "QueryEngine/Execute.h"

#include "IR/Expr.h"

class QueryRewriter {
 public:
  QueryRewriter(const std::vector<InputTableInfo>& query_infos, Executor* executor)
      : query_infos_(query_infos), executor_(executor) {}
  RelAlgExecutionUnit rewrite(const RelAlgExecutionUnit& ra_exe_unit_in) const;

  RelAlgExecutionUnit rewriteAggregateOnGroupByColumn(
      const RelAlgExecutionUnit& ra_exe_unit_in) const;

 private:
  RelAlgExecutionUnit rewriteConstrainedByIn(
      const RelAlgExecutionUnit& ra_exe_unit_in) const;

  RelAlgExecutionUnit rewriteConstrainedByInImpl(
      const RelAlgExecutionUnit& ra_exe_unit_in,
      const std::shared_ptr<hdk::ir::CaseExpr>,
      const hdk::ir::InValues*) const;

  static std::shared_ptr<hdk::ir::CaseExpr> generateCaseForDomainValues(
      const hdk::ir::InValues*);

  std::pair<bool, std::set<size_t>> is_all_groupby_exprs_are_col_var(
      const std::list<hdk::ir::ExprPtr>& groupby_exprs) const;

  std::shared_ptr<hdk::ir::CaseExpr> generateCaseExprForCountDistinctOnGroupByCol(
      hdk::ir::ExprPtr expr,
      const SQLTypeInfo& ti) const;

  const std::vector<InputTableInfo>& query_infos_;
  Executor* executor_;
  mutable std::vector<hdk::ir::ExprPtr> target_exprs_owned_;
};
