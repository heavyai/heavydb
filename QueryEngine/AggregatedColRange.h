/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    AggregatedColRange.h
 * @brief   Cache for physical column ranges. Set by the aggregator on the leaves.
 *
 */

#ifndef QUERYENGINE_AGGREGATEDCOLRANGECACHE_H
#define QUERYENGINE_AGGREGATEDCOLRANGECACHE_H

#include "ExpressionRange.h"
#include "QueryPhysicalInputsCollector.h"

#include <unordered_map>

class AggregatedColRange {
 public:
  ExpressionRange getColRange(const PhysicalInput&) const;

  void setColRange(const PhysicalInput&, const ExpressionRange&);

  const std::unordered_map<PhysicalInput, ExpressionRange>& asMap() const;

  void clear();

 private:
  std::unordered_map<PhysicalInput, ExpressionRange> cache_;
};

#endif  // QUERYENGINE_AGGREGATEDCOLRANGECACHE_H
