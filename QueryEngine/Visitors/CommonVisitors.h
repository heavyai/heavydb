/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

class StringFunctionDetector : public ScalarExprVisitor<void*> {
 public:
  static bool hasStringFunction(const Analyzer::Expr* expr) {
    StringFunctionDetector detector;
    if (expr) {
      detector.visit(expr);
    }
    return detector.has_string_oper_;
  }

 protected:
  void* visitStringOper(const Analyzer::StringOper* expr) const override {
    has_string_oper_ = true;
    return defaultResult();
  }

 private:
  mutable bool has_string_oper_{false};
};
