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

#include "RexSubQueryIdCollector.h"

RexSubQueryIdCollector::Ids RexSubQueryIdCollector::getLiveRexSubQueryIds(
    RelAlgNode const* rel_alg_node) {
  RexSubQueryIdCollector rex_sub_query_id_collector;
  rex_sub_query_id_collector.visit(rel_alg_node);
  return std::move(rex_sub_query_id_collector.ids_);
}

void RexSubQueryIdCollector::visit(RexSubQuery const* rex_sub_query) {
  ids_.insert(rex_sub_query->getId());
  RelRexDagVisitor::visit(rex_sub_query);
}
