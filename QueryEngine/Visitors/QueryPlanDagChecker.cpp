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

#include "QueryPlanDagChecker.h"

bool QueryPlanDagChecker::getCheckResult() const {
  return detect_non_supported_node_;
}

void QueryPlanDagChecker::detectNonSupportedNode(const std::string& node_tag) const {
  detect_non_supported_node_ = true;
  non_supported_node_tag_ = node_tag;
}

std::pair<bool, std::string> QueryPlanDagChecker::hasNonSupportedNodeInDag(
    const RelAlgNode* rel_alg_node,
    const RelAlgTranslator& rel_alg_translator) {
  QueryPlanDagChecker checker(rel_alg_translator);
  checker.check(rel_alg_node);
  return std::make_pair(checker.getCheckResult(), checker.getNonSupportedNodeTag());
}

void QueryPlanDagChecker::check(const RelAlgNode* rel_alg_node) {
  ExprDagVisitor::visit(rel_alg_node);
}

void QueryPlanDagChecker::visitLogicalValues(const RelLogicalValues* rel_alg_node) {
  detectNonSupportedNode("Detect RelLogicalValues node");
  return;
}

void QueryPlanDagChecker::visitTableFunction(const RelTableFunction* rel_alg_node) {
  detectNonSupportedNode("Detect RelTableFunction node");
  return;
}

void QueryPlanDagChecker::visitCompound(const RelCompound* rel_alg_node) {
  // SINGLE_VALUE / SAMPLE query
  if (rel_alg_node->isAggregate() && rel_alg_node->size() > 0) {
    for (size_t i = 0; i < rel_alg_node->size(); ++i) {
      auto target_expr = rel_alg_node->getExpr(i);
      auto agg_expr = dynamic_cast<const hdk::ir::AggExpr*>(target_expr.get());
      if (agg_expr && (agg_expr->get_aggtype() == SQLAgg::kSINGLE_VALUE ||
                       agg_expr->get_aggtype() == SQLAgg::kSAMPLE ||
                       agg_expr->get_aggtype() == SQLAgg::kAPPROX_QUANTILE)) {
        detectNonSupportedNode(
            "Detect non-supported aggregation function: "
            "SINGLE_VALUE/SAMPLE/APPROX_QUANTILE");
        return;
      }
    }
  }
  ExprDagVisitor::visitCompound(rel_alg_node);
}

void QueryPlanDagChecker::visitLogicalUnion(const RelLogicalUnion* rel_alg_node) {
  detectNonSupportedNode("Detect RelLogicalUnion node");
}

void* QueryPlanDagChecker::visitCardinality(const hdk::ir::CardinalityExpr*) const {
  detectNonSupportedNode("Detect cardinality expression");
  return nullptr;
}

void* QueryPlanDagChecker::visitConstant(const hdk::ir::Constant* constant) const {
  if (!constant->cacheable()) {
    detectNonSupportedNode("Detect non-cacheable constant");
  }
  return nullptr;
}

void* QueryPlanDagChecker::visitBinOper(const hdk::ir::BinOper* bin_oper) const {
  if (bin_oper->get_optype() == kARRAY_AT) {
    detectNonSupportedNode("Detect ARRAY_AT operation");
    return nullptr;
  }

  // Detect heavy IN-clause pattern as a deep OR tree.
  if (bin_oper->get_optype() == kOR) {
    if (deep_or_ >= 19) {
      detectNonSupportedNode("Detect heavy IN-clause having more than 20 values");
      return nullptr;
    }
    ++deep_or_;
  }

  ExprDagVisitor::visitBinOper(bin_oper);
  if (bin_oper->get_optype() == kOR) {
    --deep_or_;
  }
  return nullptr;
}

void* QueryPlanDagChecker::visitOffsetInFragment(const hdk::ir::OffsetInFragment*) const {
  detectNonSupportedNode("Detect OFFSET_IN_FRAGMENT operation");
  return nullptr;
}

void QueryPlanDagChecker::reset() {
  detect_non_supported_node_ = false;
  non_supported_node_tag_ = "";
}

const std::string& QueryPlanDagChecker::getNonSupportedNodeTag() const {
  return non_supported_node_tag_;
}
