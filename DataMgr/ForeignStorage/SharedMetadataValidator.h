/*
 * Copyright 2021 OmniSci, Inc.
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
