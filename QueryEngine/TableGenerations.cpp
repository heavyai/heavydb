#include "TableGenerations.h"

#include <glog/logging.h>

void TableGenerations::setGeneration(const uint32_t id, const TableGeneration& generation) {
  const auto it_ok = id_to_generation_.emplace(id, generation);
  CHECK(it_ok.second);
}

const TableGeneration& TableGenerations::getGeneration(const uint32_t id) const {
  const auto it = id_to_generation_.find(id);
  CHECK(it != id_to_generation_.end());
  return it->second;
}

const std::unordered_map<uint32_t, TableGeneration>& TableGenerations::asMap() const {
  return id_to_generation_;
}

void TableGenerations::clear() {
  decltype(id_to_generation_)().swap(id_to_generation_);
}
