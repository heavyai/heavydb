/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    RexSubQueryIdCollector.h
 * @brief   RexSubQueryIdCollector is a visitor class that collects all
 * RexSubQuery::getId() values for all RexSubQuery nodes. This uses sorted arrays of
 * (hash_code, handler) pairs for tree navigation.
 */

#pragma once

#include "RelRexDagVisitor.h"

#include <unordered_set>

class RexSubQueryIdCollector final : public RelRexDagVisitor {
 public:
  using RelRexDagVisitor::visit;

  using Ids = std::unordered_set<unsigned>;
  static Ids getLiveRexSubQueryIds(RelAlgNode const*);

 private:
  void visit(RexSubQuery const*) override;

  Ids ids_;
};
