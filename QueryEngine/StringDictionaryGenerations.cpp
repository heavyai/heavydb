/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StringDictionaryGenerations.h"

#include "Logger/Logger.h"

void StringDictionaryGenerations::setGeneration(const shared::StringDictKey& dict_key,
                                                const uint64_t generation) {
  dict_key_to_generation_.emplace(dict_key, generation);
}

void StringDictionaryGenerations::updateGeneration(const shared::StringDictKey& dict_key,
                                                   const uint64_t generation) {
  CHECK(dict_key_to_generation_.count(dict_key));
  dict_key_to_generation_[dict_key] = generation;
}

int64_t StringDictionaryGenerations::getGeneration(
    const shared::StringDictKey& id) const {
  const auto it = dict_key_to_generation_.find(id);
  if (it != dict_key_to_generation_.end()) {
    return it->second;
  }
  if (id.isTransientDict()) {
    return 0;
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

const std::unordered_map<shared::StringDictKey, uint64_t>&
StringDictionaryGenerations::asMap() const {
  return dict_key_to_generation_;
}

void StringDictionaryGenerations::clear() {
  decltype(dict_key_to_generation_)().swap(dict_key_to_generation_);
}
