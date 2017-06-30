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

#include "StringDictionaryGenerations.h"

#include <glog/logging.h>

void StringDictionaryGenerations::setGeneration(const uint32_t id, const size_t generation) {
  id_to_generation_.emplace(id, generation);
}

ssize_t StringDictionaryGenerations::getGeneration(const uint32_t id) const {
  const auto it = id_to_generation_.find(id);
  if (it != id_to_generation_.end()) {
    return it->second;
  }
  // This happens when the query didn't need to do any translation from string
  // to id. Return an invalid generation and StringDictionaryProxy will assert
  // the methods which require a generation (the ones which go from string to id)
  // are called on it if it has an invalid generation, only the id from string
  // direction is allowed in such cases. Once a query finishes its effective
  // execution, the generations are cleared and the string to id direction
  // isn't needed anymore, only the opposite for the returned result set.
  return -1;
}

const std::unordered_map<uint32_t, size_t>& StringDictionaryGenerations::asMap() const {
  return id_to_generation_;
}

void StringDictionaryGenerations::clear() {
  decltype(id_to_generation_)().swap(id_to_generation_);
}
