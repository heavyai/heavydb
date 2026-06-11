/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>

#include "ScalarExprVisitor.h"

class MaxRangeTableIndexVisitor : public ScalarExprVisitor<int> {
 protected:
  int visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return column->get_rte_idx();
  }

  int visitColumnVarTuple(const Analyzer::ExpressionTuple* expr_tuple) const override {
    MaxRangeTableIndexVisitor visitor;
    int max_range_table_idx = 0;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      max_range_table_idx =
          std::max(max_range_table_idx, visitor.visit(expr_component.get()));
    }
    return max_range_table_idx;
  }

  int aggregateResult(const int& aggregate, const int& next_result) const override {
    return std::max(aggregate, next_result);
  }
};

class AllRangeTableIndexVisitor : public ScalarExprVisitor<std::set<int>> {
 protected:
  std::set<int> visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return {column->get_rte_idx()};
  }

  std::set<int> visitColumnVarTuple(
      const Analyzer::ExpressionTuple* expr_tuple) const override {
    AllRangeTableIndexVisitor visitor;
    std::set<int> result;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      const auto component_rte_set = visitor.visit(expr_component.get());
      result.insert(component_rte_set.begin(), component_rte_set.end());
    }
    return result;
  }

  std::set<int> aggregateResult(const std::set<int>& aggregate,
                                const std::set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};
