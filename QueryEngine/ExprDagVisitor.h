/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RelAlgDagBuilder.h"
#include "ScalarExprVisitor.h"

// TODO: this visitor similar to RelRexDagVisitor can visit the same node
// multiple times.
class ExprDagVisitor : public ScalarExprVisitor<void*> {
 public:
  using ScalarExprVisitor::visit;

  void visit(const RelAlgNode* node) {
    if (auto agg = dynamic_cast<const RelAggregate*>(node)) {
      visitAggregate(agg);
    } else if (auto compound = dynamic_cast<const RelCompound*>(node)) {
      visitCompound(compound);
    } else if (auto filter = dynamic_cast<const RelFilter*>(node)) {
      visitFilter(filter);
    } else if (auto join = dynamic_cast<const RelJoin*>(node)) {
      visitJoin(join);
    } else if (auto deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(node)) {
      visitLeftDeepInnerJoin(deep_join);
    } else if (auto logical_union = dynamic_cast<const RelLogicalUnion*>(node)) {
      visitLogicalUnion(logical_union);
    } else if (auto values = dynamic_cast<const RelLogicalValues*>(node)) {
      visitLogicalValues(values);
    } else if (auto project = dynamic_cast<const RelProject*>(node)) {
      visitProject(project);
    } else if (auto scan = dynamic_cast<const RelScan*>(node)) {
      visitScan(scan);
    } else if (auto sort = dynamic_cast<const RelSort*>(node)) {
      visitSort(sort);
    } else if (auto table_fn = dynamic_cast<const RelTableFunction*>(node)) {
      visitTableFunction(table_fn);
    } else if (auto translated_join = dynamic_cast<const RelTranslatedJoin*>(node)) {
      visitTranslatedJoin(translated_join);
    } else {
      LOG(FATAL) << "Unsupported node type: " << node->toString();
    }
    for (size_t i = 0; i < node->inputCount(); ++i) {
      visit(node->getInput(i));
    }
  }

 protected:
  virtual void visitAggregate(const RelAggregate* agg) {
    for (auto& expr : agg->getAggs()) {
      visit(expr.get());
    }
  }

  virtual void visitCompound(const RelCompound* compound) {
    if (compound->getFilter()) {
      visit(compound->getFilter().get());
    }
    for (auto& expr : compound->getGroupByExprs()) {
      visit(expr.get());
    }
    for (auto& expr : compound->getExprs()) {
      visit(expr.get());
    }
  }

  virtual void visitFilter(const RelFilter* filter) { visit(filter->getConditionExpr()); }

  virtual void visitJoin(const RelJoin* join) { visit(join->getCondition()); }

  virtual void visitLeftDeepInnerJoin(const RelLeftDeepInnerJoin* join) {
    visit(join->getInnerCondition());
    for (size_t level = 1; level < join->inputCount(); ++level) {
      if (auto* outer_condition = join->getOuterCondition(level)) {
        visit(outer_condition);
      }
    }
  }

  virtual void visitLogicalUnion(const RelLogicalUnion*) {}

  virtual void visitLogicalValues(const RelLogicalValues* logical_values) {
    for (size_t row_idx = 0; row_idx < logical_values->getNumRows(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < logical_values->getRowsSize(); ++col_idx) {
        CHECK_EQ(logical_values->getValueExprAt(row_idx, col_idx), nullptr) << "NYI";
        // visit(logical_values->getValueExprAt(row_idx, col_idx));
      }
    }
  }

  virtual void visitProject(const RelProject* proj) {
    for (auto& expr : proj->getExprs()) {
      visit(expr.get());
    }
  }

  virtual void visitScan(const RelScan*) {}
  virtual void visitSort(const RelSort*) {}

  virtual void visitTableFunction(const RelTableFunction* table_function) {
    for (size_t i = 0; i < table_function->getTableFuncInputsSize(); ++i) {
      visit(table_function->getTableFuncInputExprAt(i));
    }
  }

  virtual void visitTranslatedJoin(const RelTranslatedJoin* translated_join) {
    visit(translated_join->getLHS());
    visit(translated_join->getRHS());
    for (auto& expr : translated_join->getFilterCond()) {
      visit(expr.get());
    }
    if (auto* outer_join_condition = translated_join->getOuterJoinCond()) {
      visit(outer_join_condition);
    }
  }

  void* visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    const_cast<ExprDagVisitor*>(this)->visit(subquery->getNode());
    return nullptr;
  }

  void* visitInSubquery(const hdk::ir::InSubquery* subquery) const override {
    const_cast<ExprDagVisitor*>(this)->visit(subquery->getNode());
    return nullptr;
  }
};
