/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
