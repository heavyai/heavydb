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
  return contain_not_supported_node_;
}

void QueryPlanDagChecker::detectNotSupportedNode() {
  contain_not_supported_node_ = true;
}

bool QueryPlanDagChecker::isNotSupportedDag(const RelAlgNode* rel_alg_node,
                                            const RelAlgTranslator& rel_alg_translator) {
  QueryPlanDagChecker checker(rel_alg_translator);
  checker.check(rel_alg_node);
  return checker.getCheckResult();
}

void QueryPlanDagChecker::check(const RelAlgNode* rel_alg_node) {
  RelRexDagVisitor::visit(rel_alg_node);
}

void QueryPlanDagChecker::visit(const RelLogicalValues* rel_alg_node) {
  detectNotSupportedNode();
  return;
}

void QueryPlanDagChecker::visit(const RelModify* rel_alg_node) {
  detectNotSupportedNode();
  return;
}

void QueryPlanDagChecker::visit(const RelTableFunction* rel_alg_node) {
  detectNotSupportedNode();
  return;
}

void QueryPlanDagChecker::visit(const RelProject* rel_alg_node) {
  if (rel_alg_node->isDeleteViaSelect() || rel_alg_node->isUpdateViaSelect() ||
      rel_alg_node->isVarlenUpdateRequired()) {
    detectNotSupportedNode();
    return;
  }
  RelRexDagVisitor::visit(rel_alg_node);
}

void QueryPlanDagChecker::visit(const RelCompound* rel_alg_node) {
  if (rel_alg_node->isDeleteViaSelect() || rel_alg_node->isUpdateViaSelect() ||
      rel_alg_node->isVarlenUpdateRequired()) {
    detectNotSupportedNode();
    return;
  }
  // SINGLE_VALUE / SAMPLE query
  if (rel_alg_node->isAggregate() && rel_alg_node->size() > 0) {
    for (size_t i = 0; i < rel_alg_node->size(); ++i) {
      auto target_expr = rel_alg_node->getTargetExpr(i);
      auto agg_expr = dynamic_cast<const RexAgg*>(target_expr);
      if (agg_expr && (agg_expr->getKind() == SQLAgg::kSINGLE_VALUE ||
                       agg_expr->getKind() == SQLAgg::kSAMPLE ||
                       agg_expr->getKind() == SQLAgg::kAPPROX_QUANTILE)) {
        detectNotSupportedNode();
        return;
      }
    }
  }
  RelRexDagVisitor::visit(rel_alg_node);
}

void QueryPlanDagChecker::visit(const RelLogicalUnion* rel_alg_node) {
  detectNotSupportedNode();
  return;
}

void QueryPlanDagChecker::visit(const RelScan* rel_alg_node) {
  if (rel_alg_node->getTableDescriptor()->storageType == StorageType::FOREIGN_TABLE) {
    detectNotSupportedNode();
    return;
  }
  RelRexDagVisitor::visit(rel_alg_node);
}

void QueryPlanDagChecker::visit(RexOperator const* rex_node) {
  // prevent too heavy IN clause containing more than 20 values to check
  if (rex_node->getOperator() == SQLOps::kOR && rex_node->size() > 20) {
    detectNotSupportedNode();
    return;
  }
  for (size_t i = 0; i < rex_node->size(); ++i) {
    if (rex_node->getOperand(i)) {
      RelRexDagVisitor::visit(rex_node->getOperand(i));
    }
  }
}

void QueryPlanDagChecker::visit(RexFunctionOperator const* rex_node) {
  if (not_supported_functions_.count(rex_node->getName())) {
    detectNotSupportedNode();
    if (rex_node->getName() == "DATETIME") {
      const auto arg = rel_alg_translator_.translateScalarRex(rex_node->getOperand(0));
      const auto arg_lit = std::dynamic_pointer_cast<Analyzer::Constant>(arg);
      if (arg_lit && !arg_lit->get_is_null() && arg_lit->get_type_info().is_string()) {
        if (*arg_lit->get_constval().stringval != "NOW"sv) {
          reset();
        }
      }
    }
  }
  if (getCheckResult()) {
    return;
  }
  for (size_t i = 0; i < rex_node->size(); ++i) {
    if (rex_node->getOperand(i)) {
      RelRexDagVisitor::visit(rex_node->getOperand(i));
    }
  }
}

void QueryPlanDagChecker::reset() {
  contain_not_supported_node_ = false;
}
