/*
 * Copyright 2024 HEAVY.AI, Inc.
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

#include "Analyzer/Analyzer.h"
#include "RexVisitor.h"
#include "ScalarExprVisitor.h"

#include <set>

class AllColumnVarsVisitor
    : public ScalarExprVisitor<std::set<const Analyzer::ColumnVar*>> {
 protected:
  std::set<const Analyzer::ColumnVar*> visitColumnVar(
      const Analyzer::ColumnVar* column) const override {
    return {column};
  }

  std::set<const Analyzer::ColumnVar*> visitColumnVarTuple(
      const Analyzer::ExpressionTuple* expr_tuple) const override {
    AllColumnVarsVisitor visitor;
    std::set<const Analyzer::ColumnVar*> result;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      const auto component_rte_set = visitor.visit(expr_component.get());
      result.insert(component_rte_set.begin(), component_rte_set.end());
    }
    return result;
  }

  std::set<const Analyzer::ColumnVar*> aggregateResult(
      const std::set<const Analyzer::ColumnVar*>& aggregate,
      const std::set<const Analyzer::ColumnVar*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

class StringOperatorDetector : public RexVisitor<void*> {
 public:
  StringOperatorDetector(SqlStringOpKind kind) {
    std::ostringstream oss;
    oss << kind;
    kind_ = oss.str();
  }

  static bool hasStringOperator(SqlStringOpKind kind, const RexScalar* expr) {
    StringOperatorDetector detector(kind);
    detector.visit(expr);
    return detector.has_string_oper_;
  }

 protected:
  void* visitOperator(const RexOperator* rex_operator) const override {
    if (auto rex_func = dynamic_cast<const RexFunctionOperator*>(rex_operator)) {
      if (rex_func->getName() == kind_) {
        has_string_oper_ = true;
        return defaultResult();
      }
    }
    for (size_t i = 0; i < rex_operator->size(); ++i) {
      visit(rex_operator->getOperand(i));
    }
    return defaultResult();
  }

 private:
  std::string kind_;
  mutable bool has_string_oper_{false};
};
