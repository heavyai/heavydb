/*
 * Copyright 2020 OmniSci, Inc.
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

/*
 * Q: Why are std::arrays used, instead of std::unordered_maps to match type_index to
 * their handlers?
 *
 * A: Since they are static variables, they should be trivially destructible. See
 * https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
 */

#include "RelRexDagVisitor.h"
#include "Shared/Logger.h"

#include <algorithm>
#include <typeindex>

template <typename T, typename... Ts>
RelRexDagVisitor::Handlers<T, sizeof...(Ts)> RelRexDagVisitor::make_handlers() {
  RelRexDagVisitor::Handlers<T, sizeof...(Ts)> handlers{
      {{std::type_index(typeid(Ts)), &RelRexDagVisitor::cast<T, Ts>}...}};
  std::sort(handlers.begin(), handlers.end());
  return handlers;
}

// RelAlgNode types

void RelRexDagVisitor::visit(RelAlgNode const* rel_alg_node) {
  // Array that pairs std::type_index(typeid(*rel_alg_node)) -> method pointer.
  static auto const handlers = make_handlers<RelAlgNode,
                                             RelAggregate,
                                             RelCompound,
                                             RelFilter,
                                             RelJoin,
                                             RelLeftDeepInnerJoin,
                                             RelLogicalUnion,
                                             RelLogicalValues,
                                             RelModify,
                                             RelProject,
                                             RelScan,
                                             RelSort,
                                             RelTableFunction>();
  static_assert(std::is_trivially_destructible_v<decltype(handlers)>);
  // Will throw std::bad_typeid if rel_alg_node == nullptr.
  auto const& type_index = std::type_index(typeid(*rel_alg_node));
  auto const itr = std::lower_bound(handlers.cbegin(), handlers.cend(), type_index);
  if (itr != handlers.cend() && itr->type_index == type_index) {
    (this->*itr->handler)(rel_alg_node);
  } else {
    LOG(FATAL) << "Unhandled RelAlgNode type: " << rel_alg_node->toString();
  }
  for (size_t i = 0; i < rel_alg_node->inputCount(); ++i) {
    visit(rel_alg_node->getInput(i));
  }
}

void RelRexDagVisitor::visit(RelCompound const* rel_compound) {
  if (rel_compound->getFilterExpr()) {
    visit(rel_compound->getFilterExpr());
  }
}

void RelRexDagVisitor::visit(RelFilter const* rel_filter) {
  visit(rel_filter->getCondition());
}

void RelRexDagVisitor::visit(RelJoin const* rel_join) {
  visit(rel_join->getCondition());
}

void RelRexDagVisitor::visit(RelLeftDeepInnerJoin const* rel_left_deep_inner_join) {
  visit(rel_left_deep_inner_join->getInnerCondition());
  for (size_t level = 1; level < rel_left_deep_inner_join->inputCount(); ++level) {
    visit(rel_left_deep_inner_join->getOuterCondition(level));
  }
}

void RelRexDagVisitor::visit(RelLogicalValues const* rel_logical_values) {
  for (size_t row_idx = 0; row_idx < rel_logical_values->getNumRows(); ++row_idx) {
    for (size_t col_idx = 0; col_idx < rel_logical_values->getRowsSize(); ++col_idx) {
      visit(rel_logical_values->getValueAt(row_idx, col_idx));
    }
  }
}

void RelRexDagVisitor::visit(RelProject const* rel_projection) {
  for (size_t i = 0; i < rel_projection->size(); ++i) {
    visit(rel_projection->getProjectAt(i));
  }
}

void RelRexDagVisitor::visit(RelTableFunction const* rel_table_function) {
  for (size_t i = 0; i < rel_table_function->getTableFuncInputsSize(); ++i) {
    visit(rel_table_function->getTableFuncInputAt(i));
  }
}

// RexScalar types

void RelRexDagVisitor::visit(RexScalar const* rex_scalar) {
  // Array that pairs std::type_index(typeid(*rex_scalar)) -> method pointer.
  static auto const handlers = make_handlers<RexScalar,
                                             RexAbstractInput,
                                             RexCase,
                                             RexFunctionOperator,
                                             RexInput,
                                             RexLiteral,
                                             RexOperator,
                                             RexRef,
                                             RexSubQuery,
                                             RexWindowFunctionOperator>();
  static_assert(std::is_trivially_destructible_v<decltype(handlers)>);
  // Will throw std::bad_typeid if rex_scalar == nullptr.
  auto const& type_index = std::type_index(typeid(*rex_scalar));
  auto const itr = std::lower_bound(handlers.cbegin(), handlers.cend(), type_index);
  if (itr != handlers.cend() && itr->type_index == type_index) {
    (this->*itr->handler)(rex_scalar);
  } else {
    LOG(FATAL) << "Unhandled RexScalar type: " << rex_scalar->toString();
  }
}

void RelRexDagVisitor::visit(
    RexWindowFunctionOperator const* rex_window_function_operator) {
  for (const auto& partition_key : rex_window_function_operator->getPartitionKeys()) {
    visit(partition_key.get());
  }
  for (const auto& order_key : rex_window_function_operator->getOrderKeys()) {
    visit(order_key.get());
  }
}

void RelRexDagVisitor::visit(RexCase const* rex_case) {
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    visit(rex_case->getWhen(i));
    visit(rex_case->getThen(i));
  }
  if (rex_case->getElse()) {
    visit(rex_case->getElse());
  }
}

void RelRexDagVisitor::visit(RexFunctionOperator const* rex_function_operator) {
  for (size_t i = 0; i < rex_function_operator->size(); ++i) {
    visit(rex_function_operator->getOperand(i));
  }
}

void RelRexDagVisitor::visit(RexOperator const* rex_operator) {
  for (size_t i = 0; i < rex_operator->size(); ++i) {
    visit(rex_operator->getOperand(i));
  }
}

void RelRexDagVisitor::visit(RexSubQuery const* rex_sub_query) {
  visit(rex_sub_query->getRelAlg());
}
