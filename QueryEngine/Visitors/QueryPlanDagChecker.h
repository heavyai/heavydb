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

#include "QueryEngine/ExprDagVisitor.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgTranslator.h"

class QueryPlanDagChecker final : public ExprDagVisitor {
 public:
  QueryPlanDagChecker(const RelAlgTranslator& rel_alg_translator)
      : detect_non_supported_node_(false)
      , non_supported_node_tag_("")
      , rel_alg_translator_(rel_alg_translator) {}

  static std::pair<bool, std::string> hasNonSupportedNodeInDag(
      const RelAlgNode* rel_alg_node,
      const RelAlgTranslator& rel_alg_translator);
  void check(const RelAlgNode*);
  void detectNonSupportedNode(const std::string& node_tag) const;
  void reset();
  bool getCheckResult() const;
  std::string const& getNonSupportedNodeTag() const;

 private:
  void visitLogicalValues(const RelLogicalValues*) override;
  void visitTableFunction(const RelTableFunction*) override;
  void visitCompound(const RelCompound*) override;
  void visitLogicalUnion(const RelLogicalUnion*) override;

  void* visitCardinality(const hdk::ir::CardinalityExpr* cardinality) const override;
  void* visitConstant(const hdk::ir::Constant* constant) const override;
  void* visitBinOper(const hdk::ir::BinOper* bin_oper) const override;
  void* visitOffsetInFragment(const hdk::ir::OffsetInFragment*) const override;

  mutable bool detect_non_supported_node_;
  mutable std::string non_supported_node_tag_;
  const RelAlgTranslator& rel_alg_translator_;
  mutable int deep_or_ = 0;
};
