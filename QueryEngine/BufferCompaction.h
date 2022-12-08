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
 * @file    BufferCompaction.h
 * @brief   Macros and functions for groupby buffer compaction
 *
 */

#ifndef BUFFER_COMPACTION_H
#define BUFFER_COMPACTION_H

#include <cstdint>
#include "../Shared/funcannotations.h"

#ifndef __CUDACC__
#include <algorithm>
#endif

constexpr int8_t MAX_BYTE_WIDTH_SUPPORTED = 8;

#ifndef __CUDACC__
inline unsigned compact_byte_width(unsigned qw, unsigned low_bound) {
  return std::max(qw, low_bound);
}
#endif

template <typename T>
FORCE_INLINE HOST DEVICE T align_to_int64(T addr) {
  addr += sizeof(int64_t) - 1;
  return (T)(((uint64_t)addr >> 3) << 3);
}

// Return nearest multiple of N for val rounding up.
// Example: align_to<8>(10) = 16.
template <size_t N, typename T>
FORCE_INLINE HOST DEVICE T align_to(T const val) {
  constexpr T mask = static_cast<T>(N - 1);
  static_assert(N && (N & mask) == 0, "N must be a power of 2.");
  return (val + mask) & ~mask;
}

#endif /* BUFFER_COMPACTION_H */
