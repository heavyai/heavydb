/*
 * Copyright 2022 HEAVY.AI, Inc.
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
  QueryPlanDagChecker()
      : detect_non_supported_node_(false)
      , non_supported_node_tag_("")
      , non_supported_functions_(getNonSupportedFunctionsList()) {}
  static std::unordered_set<std::string> getNonSupportedFunctionsList() {
    std::unordered_set<std::string> non_supported_functions;
    non_supported_functions.emplace("CURRENT_USER");
    non_supported_functions.emplace("CURRENT_TIME");
    non_supported_functions.emplace("CURRENT_DATE");
    non_supported_functions.emplace("CURRENT_TIMESTAMP");
    non_supported_functions.emplace("CARDINALITY");
    non_supported_functions.emplace("ARRAY_LENGTH");
    non_supported_functions.emplace("ITEM");
    non_supported_functions.emplace("NOW");
    non_supported_functions.emplace("SIGN");
    non_supported_functions.emplace("OFFSET_IN_FRAGMENT");
    non_supported_functions.emplace("DATETIME");
    return non_supported_functions;
  }

  static std::pair<bool, std::string> hasNonSupportedNodeInDag(
      const RelAlgNode* rel_alg_node);
  void check(const RelAlgNode*);
  void detectNonSupportedNode(const std::string& node_tag);
  void reset();
  bool getCheckResult() const;
  std::string const& getNonSupportedNodeTag() const;

 private:
  void visit(const RelLogicalValues*) override;
  void visit(const RelModify*) override;
  void visit(const RelProject*) override;
  void visit(const RelScan*) override;
  void visit(const RelCompound*) override;
  void visit(const RelLogicalUnion*) override;

  void visit(const RexFunctionOperator*) override;
  void visit(const RexOperator*) override;

  bool detect_non_supported_node_;
  std::string non_supported_node_tag_;
  const std::unordered_set<std::string> non_supported_functions_;
};
