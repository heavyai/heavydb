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

#include <llvm/IR/Value.h>
#include <boost/optional.hpp>
#include <list>
#include <memory>
#include <vector>

#include "IR/Expr.h"
#include "RelAlgExecutionUnit.h"

namespace hdk::ir {

class Expr;

class InValues;

}  // namespace hdk::ir

class InputColDescriptor;

// Rewrites an OR tree where leaves are equality compare against literals.
hdk::ir::ExprPtr rewrite_expr(const hdk::ir::Expr*);

// Rewrites array elements that are strings to be dict encoded transient literals
hdk::ir::ExprPtr rewrite_array_elements(const hdk::ir::Expr*);

// Rewrite a FunctionOper to an AND between a BinOper and the FunctionOper if the
// FunctionOper is supported for overlaps joins
struct OverlapsJoinConjunction {
  std::list<hdk::ir::ExprPtr> quals;
  std::list<hdk::ir::ExprPtr> join_quals;
};

boost::optional<OverlapsJoinConjunction> rewrite_overlaps_conjunction(
    const hdk::ir::ExprPtr expr);

std::list<hdk::ir::ExprPtr> strip_join_covered_filter_quals(
    const std::list<hdk::ir::ExprPtr>& quals,
    const JoinQualsPerNestingLevel& join_quals);

hdk::ir::ExprPtr fold_expr(const hdk::ir::Expr*);

bool self_join_not_covered_by_left_deep_tree(const hdk::ir::ColumnVar* lhs,
                                             const hdk::ir::ColumnVar* rhs,
                                             const int max_rte_covered);

const int get_max_rte_scan_table(
    std::unordered_map<int, llvm::Value*>& scan_idx_to_hash_pos);

#endif  // QUERYENGINE_EXPRESSIONREWRITE_H
