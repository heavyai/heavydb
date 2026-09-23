/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RelRexDagVisitor.h"

class SQLOperatorDetector final : public RelRexDagVisitor {
 public:
  using RelRexDagVisitor::visit;

  static bool detect(RelAlgNode const* rel_alg_node, SQLOps target_op);

 private:
  void visit(RexOperator const* op) override;

  bool has_target_op_{false};
  SQLOps target_op_;
};