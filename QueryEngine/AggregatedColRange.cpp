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

#include "AggregatedColRange.h"

ExpressionRange AggregatedColRange::getColRange(const PhysicalInput& phys_input) const {
  const auto it = cache_.find(phys_input);
  CHECK(it != cache_.end());
  return it->second;
}

void AggregatedColRange::setColRange(const PhysicalInput& phys_input, const ExpressionRange& expr_range) {
  const auto it_ok = cache_.emplace(phys_input, expr_range);
  CHECK(it_ok.second);
}

const std::unordered_map<PhysicalInput, ExpressionRange>& AggregatedColRange::asMap() const {
  return cache_;
}

void AggregatedColRange::clear() {
  decltype(cache_)().swap(cache_);
}
