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

  hdk::ir::ExprPtr translateScalarRex(const RexScalar* rex) const;

  static hdk::ir::ExprPtr translateAggregateRex(
      const RexAgg* rex,
      const std::vector<hdk::ir::ExprPtr>& scalar_sources,
      bool bigint_count);

  hdk::ir::ExprPtr translateColumnRefs(const hdk::ir::Expr* expr) const;

  static hdk::ir::ExprPtr translateLiteral(const RexLiteral*);

 private:
  hdk::ir::ExprPtr translateScalarSubquery(const RexSubQuery*) const;

  hdk::ir::ExprPtr translateInput(const RexInput*) const;

  hdk::ir::ExprPtr translateUoper(const RexOperator*) const;

  hdk::ir::ExprPtr translateInOper(const RexOperator*) const;

  hdk::ir::ExprPtr getInIntegerSetExpr(hdk::ir::ExprPtr arg,
                                       const ResultSet& val_set) const;

  hdk::ir::ExprPtr translateOper(const RexOperator*) const;

  hdk::ir::ExprPtr translateOverlapsOper(const RexOperator*) const;

  hdk::ir::ExprPtr translateCase(const RexCase*) const;

  hdk::ir::ExprPtr translateWidthBucket(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateLike(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateRegexp(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateLikely(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateUnlikely(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateExtract(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateDateadd(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateDatePlusMinus(const RexOperator*) const;

  hdk::ir::ExprPtr translateDatediff(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateDatepart(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateLength(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateKeyForString(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateSampleRatio(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateCurrentUser(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateLower(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateCardinality(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateItem(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateCurrentDate() const;

  hdk::ir::ExprPtr translateCurrentTime() const;

  hdk::ir::ExprPtr translateCurrentTimestamp() const;

  hdk::ir::ExprPtr translateDatetime(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateHPTLiteral(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateAbs(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateSign(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateOffsetInFragment() const;

  hdk::ir::ExprPtr translateArrayFunction(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateFunction(const RexFunctionOperator*) const;

  hdk::ir::ExprPtr translateWindowFunction(const RexWindowFunctionOperator*) const;

  hdk::ir::ExprPtrVector translateFunctionArgs(const RexFunctionOperator*) const;

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

inline auto func_resolve = [](auto funcname, auto&&... strlist) {
  return ((funcname == strlist) || ...);
};

using namespace std::literals;

#endif  // QUERYENGINE_RELALGTRANSLATOR_H
