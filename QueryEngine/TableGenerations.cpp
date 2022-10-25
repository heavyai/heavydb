/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "TableGenerations.h"
#include "Logger/Logger.h"

void TableGenerations::setGeneration(const shared::TableKey& table_key,
                                     const TableGeneration& generation) {
  const auto it_ok = table_key_to_generation_.emplace(table_key, generation);
  CHECK(it_ok.second);
}

const TableGeneration& TableGenerations::getGeneration(
    const shared::TableKey& table_key) const {
  const auto it = table_key_to_generation_.find(table_key);
  CHECK(it != table_key_to_generation_.end());
  return it->second;
}

const std::unordered_map<shared::TableKey, TableGeneration>& TableGenerations::asMap()
    const {
  return table_key_to_generation_;
}

void TableGenerations::clear() {
  decltype(table_key_to_generation_)().swap(table_key_to_generation_);
}
