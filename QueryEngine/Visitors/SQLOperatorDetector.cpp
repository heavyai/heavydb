/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SQLOperatorDetector.h"

bool SQLOperatorDetector::detect(RelAlgNode const* rel_alg_node, SQLOps target_op) {
  SQLOperatorDetector visitor;
  visitor.target_op_ = target_op;
  visitor.visit(rel_alg_node);
  return visitor.has_target_op_;
}

void SQLOperatorDetector::visit(RexOperator const* op) {
  if (op->getOperator() == target_op_) {
    has_target_op_ = true;
  }
  RelRexDagVisitor::visit(op);
}
