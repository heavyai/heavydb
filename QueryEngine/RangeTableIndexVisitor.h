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

#pragma once

#include <unordered_set>

#include "ScalarExprVisitor.h"

class MaxRangeTableIndexVisitor : public ScalarExprVisitor<int> {
 protected:
  virtual int visitColumnVar(const Analyzer::ColumnVar* column) const override { return column->get_rte_idx(); }

  virtual int visitColumnVarTuple(const Analyzer::ExpressionTuple* expr_tuple) const override {
    MaxRangeTableIndexVisitor visitor;
    int max_range_table_idx = 0;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      max_range_table_idx = std::max(max_range_table_idx, visitor.visit(expr_component.get()));
    }
    return max_range_table_idx;
  }

  virtual int aggregateResult(const int& aggregate, const int& next_result) const override {
    return std::max(aggregate, next_result);
  }
};

class AllRangeTableIndexVisitor : public ScalarExprVisitor<std::unordered_set<int>> {
 protected:
  virtual std::unordered_set<int> visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return {column->get_rte_idx()};
  }

  virtual std::unordered_set<int> visitColumnVarTuple(const Analyzer::ExpressionTuple* expr_tuple) const override {
    AllRangeTableIndexVisitor visitor;
    std::unordered_set<int> result;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      const auto component_rte_set = visitor.visit(expr_component.get());
      result.insert(component_rte_set.begin(), component_rte_set.end());
    }
    return result;
  }

  virtual std::unordered_set<int> aggregateResult(const std::unordered_set<int>& aggregate,
                                                  const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};
