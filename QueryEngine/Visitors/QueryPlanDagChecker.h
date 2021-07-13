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

#pragma once

#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "RelRexDagVisitor.h"

#include <unordered_set>

class QueryPlanDagChecker final : public RelRexDagVisitor {
 public:
  QueryPlanDagChecker(const RelAlgTranslator& rel_alg_translator)
      : contain_not_supported_node_(false)
      , rel_alg_translator_(rel_alg_translator)
      , not_supported_functions_(getNotSupportedFunctionsList()) {}
  static std::unordered_set<std::string> getNotSupportedFunctionsList() {
    std::unordered_set<std::string> not_supported_functions;
    not_supported_functions.emplace("CURRENT_USER");
    not_supported_functions.emplace("CARDINALITY");
    not_supported_functions.emplace("ARRAY_LENGTH");
    not_supported_functions.emplace("ITEM");
    not_supported_functions.emplace("NOW");
    not_supported_functions.emplace("SIGN");
    not_supported_functions.emplace("OFFSET_IN_FRAGMENT");
    not_supported_functions.emplace("DATETIME");
    return not_supported_functions;
  }

  static bool isNotSupportedDag(const RelAlgNode* rel_alg_node,
                                const RelAlgTranslator& rel_alg_translator);
  void visit(const RelAlgNode*);
  void detectNotSupportedNode();
  void reset();
  bool getCheckResult() const;

 private:
  void visit(const RelLogicalValues*) override;
  void visit(const RelModify*) override;
  void visit(const RelTableFunction*) override;
  void visit(const RelProject*) override;
  void visit(const RelScan*) override;
  void visit(const RelCompound*) override;
  void visit(const RelLogicalUnion*) override;

  void visit(const RexFunctionOperator*) override;
  void visit(const RexOperator*) override;

  bool contain_not_supported_node_;
  const RelAlgTranslator& rel_alg_translator_;
  const std::unordered_set<std::string> not_supported_functions_;
};
