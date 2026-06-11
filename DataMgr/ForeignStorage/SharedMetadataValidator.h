/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace foreign_storage {
template <typename V, std::enable_if_t<std::is_integral<V>::value, int> = 0>
inline V get_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V get_null_value() {
  return inline_fp_null_value<V>();
}

template <typename D, std::enable_if_t<std::is_integral<D>::value, int> = 0>
inline std::pair<D, D> get_min_max_bounds() {
  static_assert(std::is_signed<D>::value,
                "'get_min_max_bounds' is only valid for signed types");
  return {get_null_value<D>() + 1, std::numeric_limits<D>::max()};
}

template <typename D, std::enable_if_t<std::is_floating_point<D>::value, int> = 0>
inline std::pair<D, D> get_min_max_bounds() {
  return {std::numeric_limits<D>::lowest(), std::numeric_limits<D>::max()};
}
}  // namespace foreign_storage
