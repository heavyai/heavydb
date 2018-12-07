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

#ifndef QUERYENGINE_EXPRESSIONREWRITE_H
#define QUERYENGINE_EXPRESSIONREWRITE_H

#include <boost/optional.hpp>
#include <list>
#include <memory>
#include <vector>

#include "Analyzer/Analyzer.h"
#include "RelAlgExecutionUnit.h"

namespace Analyzer {

class Expr;

class InValues;

}  // namespace Analyzer

class InputColDescriptor;

// Rewrites an OR tree where leaves are equality compare against literals.
Analyzer::ExpressionPtr rewrite_expr(const Analyzer::Expr*);

// Rewrites array elements that are strings to be dict encoded transient literals
Analyzer::ExpressionPtr rewrite_array_elements(const Analyzer::Expr*);

// Rewrite a FunctionOper to an AND between a BinOper and the FunctionOper if the
// FunctionOper is supported for overlaps joins
struct OverlapsJoinConjunction {
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
};

boost::optional<OverlapsJoinConjunction> rewrite_overlaps_conjunction(
    const std::shared_ptr<Analyzer::Expr> expr);

std::list<std::shared_ptr<Analyzer::Expr>> strip_join_covered_filter_quals(
    const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
    const JoinQualsPerNestingLevel& join_quals);

std::shared_ptr<Analyzer::Expr> fold_expr(const Analyzer::Expr*);

#endif  // QUERYENGINE_EXPRESSIONREWRITE_H
