/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/ExprDagVisitor.h"

class InExprDetector : public ExprDagVisitor {
 public:
  void* visitInValues(const hdk::ir::InValues* in_values) const override {
    has_in_ = true;
    return nullptr;
  }

  void* visitInIntegerSet(const hdk::ir::InIntegerSet* in_integer_set) const override {
    has_in_ = true;
    return nullptr;
  }

  void* visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    has_in_ = true;
    return nullptr;
  }

  static bool detect(const RelAlgNode* node) {
    InExprDetector detector;
    detector.visit(node);
    return detector.has_in_;
  }

 protected:
  mutable bool has_in_ = false;
};
