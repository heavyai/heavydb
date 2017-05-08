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

#include "../Analyzer/Analyzer.h"
#include "../QueryEngine/Execute.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"

class QueryRewriter {
 public:
  QueryRewriter(const RelAlgExecutionUnit& ra_exe_unit,
                const std::vector<InputTableInfo>& query_infos,
                const Executor* executor,
                const Planner::Plan* plan)
      : ra_exe_unit_(ra_exe_unit), query_infos_(query_infos), executor_(executor), plan_(plan){};
  RelAlgExecutionUnit rewrite() const;

 private:
  RelAlgExecutionUnit rewriteConstrainedByIn() const;
  static std::shared_ptr<Analyzer::CaseExpr> generateCaseForDomainValues(const Analyzer::InValues*);
  RelAlgExecutionUnit rewriteConstrainedByIn(const std::shared_ptr<Analyzer::CaseExpr>,
                                             const Analyzer::InValues*) const;

  const RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<InputTableInfo>& query_infos_;
  const Executor* executor_;
  // TODO(alex): artifacts of the plan based interface below, remove.
  const Planner::Plan* plan_;
  mutable std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;
};
