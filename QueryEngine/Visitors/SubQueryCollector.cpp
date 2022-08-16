/*
 * Copyright 2020 OmniSci, Inc.
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

#include "SubQueryCollector.h"

std::unordered_set<const RelAlgNode*> SubQueryCollector::getLiveSubQueries(
    RelAlgNode const* rel_alg_node) {
  SubQueryCollector collector;
  collector.visit(rel_alg_node);
  return std::move(collector.subqueries_);
}

void* SubQueryCollector::visitScalarSubquery(
    const hdk::ir::ScalarSubquery* subquery) const {
  subqueries_.insert(subquery->getNode());
  return ExprDagVisitor::visitScalarSubquery(subquery);
}

void* SubQueryCollector::visitInSubquery(const hdk::ir::InSubquery* in_subquery) const {
  subqueries_.insert(in_subquery->getNode());
  return ExprDagVisitor::visitInSubquery(in_subquery);
}
