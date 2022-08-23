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

#ifndef QUERYENGINE_RELALGTRANSLATOR_H
#define QUERYENGINE_RELALGTRANSLATOR_H

#include "Execute.h"
#include "RelAlgDagBuilder.h"

#include <ctime>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hdk::ir {

class Expr;

}  // namespace hdk::ir

namespace Catalog_Namespace {

class Catalog;
class SessionInfo;

}  // namespace Catalog_Namespace

class RelAlgTranslator {
 public:
  RelAlgTranslator(const Executor* executor,
                   const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                   const std::vector<JoinType>& join_types,
                   const time_t now,
                   const bool just_explain);
  RelAlgTranslator(const Config& config, const time_t now, const bool just_explain);

  hdk::ir::ExprPtr normalize(const hdk::ir::Expr* expr) const;

  hdk::ir::ExprPtr translateInSubquery(
      hdk::ir::ExprPtr lhs,
      std::shared_ptr<const ExecutionResult> result) const;

 private:
  hdk::ir::ExprPtr getInIntegerSetExpr(hdk::ir::ExprPtr arg,
                                       const ResultSet& val_set) const;
  const Executor* executor_;
  const Config& config_;
  const std::unordered_map<const RelAlgNode*, int> input_to_nest_level_;
  const std::vector<JoinType> join_types_;
  time_t now_;
  const bool just_explain_;
  const bool for_dag_builder_;
};

struct QualsConjunctiveForm {
  const std::list<hdk::ir::ExprPtr> simple_quals;
  const std::list<hdk::ir::ExprPtr> quals;
};

QualsConjunctiveForm qual_to_conjunctive_form(const hdk::ir::ExprPtr qual_expr);

std::vector<hdk::ir::ExprPtr> qual_to_disjunctive_form(const hdk::ir::ExprPtr& qual_expr);

#endif  // QUERYENGINE_RELALGTRANSLATOR_H
