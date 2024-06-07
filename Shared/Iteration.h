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

#pragma once
#include <functional>
#include <vector>

namespace shared {
inline void execute_over_contiguous_indices(
    const std::vector<size_t>& indices,
    std::function<void(const size_t, const size_t)> to_execute) {
  size_t start_pos = 0;

  while (start_pos < indices.size()) {
    size_t end_pos = indices.size();
    for (size_t i = start_pos + 1; i < indices.size(); ++i) {
      if (indices[i] != indices[i - 1] + 1) {
        end_pos = i;
        break;
      }
    }
    to_execute(start_pos, end_pos);
    start_pos = end_pos;
  }
}

}  // namespace shared
