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