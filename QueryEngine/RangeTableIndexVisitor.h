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

#include "ScalarExprVisitor.h"

class RangeTableIndexVisitor : public ScalarExprVisitor<int> {
 protected:
  virtual int visitColumnVar(const Analyzer::ColumnVar* column) const override { return column->get_rte_idx(); }

  virtual int aggregateResult(const int& aggregate, const int& next_result) const override {
    return std::max(aggregate, next_result);
  }
};
