/*
 * Copyright 2020 OmniSci, Inc.
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

#ifndef OMNISCI_QUERYHINT_H
#define OMNISCI_QUERYHINT_H

#include <optional>

#include <boost/algorithm/string.hpp>

#include "ThriftHandler/CommandLineOptions.h"

struct QueryHint {
  // for each hint "H", we represent it as optional instance
  // and consider a user is given H via an input query
  // iff H has a valid value, i.e., H.get() returns a valid hint value
  // and at the same time H.has_value() returns true
  // otherwise, i.e., H.has_value() returns false, we consider
  // the H is not given from the user
  QueryHint() : registered_hint(OMNISCI_SUPPORTED_HINT_CLASS.size(), false) {}

  QueryHint& operator=(const QueryHint& other) {
    cpu_mode = other.cpu_mode;
    overlaps_bucket_threshold = other.overlaps_bucket_threshold;
    overlaps_max_size = other.overlaps_max_size;
    overlaps_allow_gpu_build = other.overlaps_allow_gpu_build;
    overlaps_no_cache = other.overlaps_no_cache;
    overlaps_keys_per_bin = other.overlaps_keys_per_bin;
    registered_hint = other.registered_hint;
    return *this;
  }

  QueryHint(const QueryHint& other) {
    cpu_mode = other.cpu_mode;
    overlaps_bucket_threshold = other.overlaps_bucket_threshold;
    overlaps_max_size = other.overlaps_max_size;
    overlaps_allow_gpu_build = other.overlaps_allow_gpu_build;
    overlaps_no_cache = other.overlaps_no_cache;
    overlaps_keys_per_bin = other.overlaps_keys_per_bin;
    registered_hint = other.registered_hint;
  }

  // general query execution
  bool cpu_mode;

  // overlaps hash join
  double overlaps_bucket_threshold;  // defined in "OverlapsJoinHashTable.h"
  size_t overlaps_max_size;
  bool overlaps_allow_gpu_build;
  bool overlaps_no_cache;
  double overlaps_keys_per_bin;

  std::unordered_map<std::string, size_t> OMNISCI_SUPPORTED_HINT_CLASS = {
      {"cpu_mode", 0},
      {"overlaps_bucket_threshold", 1},
      {"overlaps_max_size", 2},
      {"overlaps_allow_gpu_build", 3},
      {"overlaps_no_cache", 4},
      {"overlaps_keys_per_bin", 5}};

  std::vector<bool> registered_hint;

  static QueryHint defaults() { return QueryHint(); }

 public:
  bool isAnyQueryHintDelivered() const {
    for (auto& kv : OMNISCI_SUPPORTED_HINT_CLASS) {
      if (registered_hint[kv.second]) {
        return true;
      }
    }
    return false;
  }

  void registerHint(const std::string& hint_name) {
    const auto hint_class = getHintClass(hint_name);
    if (hint_class >= 0 && hint_class < OMNISCI_SUPPORTED_HINT_CLASS.size()) {
      registered_hint[hint_class] = true;
    }
  }

  void registerHint(const size_t hint_class) {
    if (hint_class >= 0 && hint_class < OMNISCI_SUPPORTED_HINT_CLASS.size()) {
      registered_hint[hint_class] = true;
    }
  }

  const bool isHintRegistered(const std::string& hint_name) const {
    const auto hint_class = getHintClass(hint_name);
    if (hint_class >= 0 && hint_class < OMNISCI_SUPPORTED_HINT_CLASS.size()) {
      return registered_hint[hint_class];
    }
    return false;
  }

  const bool isHintRegistered(const size_t hint_class) const {
    if (hint_class >= 0 && hint_class < OMNISCI_SUPPORTED_HINT_CLASS.size()) {
      return registered_hint[hint_class];
    }
    return false;
  }

  const size_t getHintClass(const std::string& hint_name) const {
    const auto lowered_hint_name = boost::algorithm::to_lower_copy(hint_name);
    auto it = OMNISCI_SUPPORTED_HINT_CLASS.find(lowered_hint_name);
    if (it != OMNISCI_SUPPORTED_HINT_CLASS.end()) {
      return it->second;
    }
    return SIZE_MAX;
  }
};

#endif  // OMNISCI_QUERYHINT_H
