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

/**
 * @file    AggMode.h
 * @brief   Calculate statistical mode as an aggregate function.
 *
 */

#pragma once

#include <algorithm>
#include <mutex>
#include <optional>
#include "ThirdParty/robin_hood/robin_hood.h"

class AggMode {
 public:
  using Value = int64_t;
  using Count = uint64_t;
  using Map = robin_hood::unordered_map<Value, Count>;
  struct ByCount {
    bool operator()(Map::value_type const& a, Map::value_type const& b) const {
      return a.second < b.second;
    }
  };
  void add(Value const value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const [itr, emplaced] = map_.emplace(value, 1u);
    if (!emplaced) {
      ++itr->second;
    }
  }
  void reduce(AggMode&& rhs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_.size() < rhs.map_.size()) {  // Loop over the smaller map
      rhs.reduceMap(map_);
      map_ = std::move(rhs.map_);
    } else {
      reduceMap(rhs.map_);
    }
  }
  std::optional<Value> mode() const {
    // In case of ties, any max element may be chosen.
    auto const itr = std::max_element(map_.begin(), map_.end(), ByCount{});
    return itr == map_.end() ? std::nullopt : std::make_optional(itr->first);
  }

 private:
  void reduceMap(Map const& map) {
    for (Map::value_type const& pair : map) {
      auto const [itr, emplaced] = map_.emplace(pair);
      if (!emplaced) {
        itr->second += pair.second;
      }
    }
  }
  Map map_;
  std::mutex mutex_;
};
