/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ScalarExprVisitor.h"
#include "Shared/DbObjectKeys.h"

#include <unordered_set>

class UsedColumnsVisitor
    : public ScalarExprVisitor<std::unordered_set<shared::ColumnKey>> {
 protected:
  std::unordered_set<shared::ColumnKey> visitColumnVar(
      const Analyzer::ColumnVar* column) const override {
    return {column->getColumnKey()};
  }

  std::unordered_set<shared::ColumnKey> aggregateResult(
      const std::unordered_set<shared::ColumnKey>& aggregate,
      const std::unordered_set<shared::ColumnKey>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};
