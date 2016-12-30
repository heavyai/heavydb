/*
 * @file    AggregatedColRange.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Cache for physical column ranges. Set by the aggregator on the leaves.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_AGGREGATEDCOLRANGECACHE_H
#define QUERYENGINE_AGGREGATEDCOLRANGECACHE_H

#include "QueryPhysicalInputsCollector.h"
#include "ExpressionRange.h"

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
