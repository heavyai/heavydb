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

#ifndef RESULTSETTESTUTILS_H
#define RESULTSETTESTUTILS_H

#include "../QueryEngine/TargetValue.h"
#include "../Shared/TargetInfo.h"

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

void fill_one_entry_baseline(int64_t* value_slots,
                             const int64_t v,
                             const std::vector<TargetInfo>& target_infos,
                             const bool empty = false,
                             const bool null_val = false);

size_t get_slot_count(const std::vector<TargetInfo>& target_infos);

std::unordered_map<size_t, size_t> get_slot_to_target_mapping(const std::vector<TargetInfo>& target_infos);

template <class T>
inline T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

#endif  // RESULTSETTESTUTILS_H
