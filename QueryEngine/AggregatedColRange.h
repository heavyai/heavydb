/*
 * Copyright 2017 MapD Technologies, Inc.
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
