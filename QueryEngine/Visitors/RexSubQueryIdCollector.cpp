/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
