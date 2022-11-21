/*
 * Copyright 2022 HEAVY.AI, Inc.
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
