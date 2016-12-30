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
