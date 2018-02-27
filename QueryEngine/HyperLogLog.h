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

/**
 * @file    HyperLogLog.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Functions used to work with HyperLogLog records.
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_HYPERLOGLOG_H
#define QUERYENGINE_HYPERLOGLOG_H

#include "CountDistinctDescriptor.h"

#include <cmath>

inline double hll_alpha(const size_t m) {
  double m_square = m * m;
  switch (m) {
    case 16:
      return 0.673 * m_square;
    case 32:
      return 0.697 * m_square;
    case 64:
      return 0.709 * m_square;
    default:
      break;
  }
  return (0.7213 / (1.0 + 1.079 / m)) * m_square;
}

template <class T>
inline size_t hll_size(const T* M, const size_t bitmap_sz_bits) {
  size_t m = 1 << bitmap_sz_bits;
  double sum{0};
  for (size_t i = 0; i < m; i++) {
    sum += 1.0 / (1ULL << M[i]);
  }
  auto estimate = hll_alpha(m) / sum;
  if (estimate <= 2.5 * m) {
    uint32_t zeros = 0;
    for (uint32_t i = 0; i < m; i++) {
      if (M[i] == 0) {
        zeros++;
      }
    }
    if (zeros != 0) {
      estimate = m * log(static_cast<double>(m) / zeros);
    }
  }
  // No correction for large estimates since we're using 64-bit hashes.
  return estimate;
}

template <class T1, class T2>
inline void hll_unify(T1* lhs, T2* rhs, const size_t m) {
  for (size_t r = 0; r < m; ++r) {
    rhs[r] = lhs[r] = std::max(static_cast<int8_t>(lhs[r]), static_cast<int8_t>(rhs[r]));
  }
}

inline int hll_size_for_rate(const int err_percent) {
  double err_rate{static_cast<double>(err_percent) / 100.0};
  double k = ceil(2 * log2(1.04 / err_rate));
  return std::min(16, std::max(static_cast<int>(k), 1));
}

extern int g_hll_precision_bits;

#endif  // QUERYENGINE_HYPERLOGLOG_H
