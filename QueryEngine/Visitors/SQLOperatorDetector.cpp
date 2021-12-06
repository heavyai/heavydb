/*
 * Copyright 2021 OmniSci, Inc.
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
