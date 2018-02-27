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

inline double get_alpha(const size_t m) {
  switch (m) {
    case 16:
      return 0.673;
    case 32:
      return 0.697;
    case 64:
      return 0.709;
    default:
      break;
  }
  double alpha = 0.7213 / (1 + 1.079 / m);
  return alpha;
}

inline double get_beta(const uint32_t zeros) {
  // Using polynomial regression terms found in LogLog-Beta paper and Redis
  double zl = log(zeros + 1);
  double beta = -0.370393911 * zeros + 0.070471823 * zl + 0.17393686 * pow(zl, 2) + 0.16339839 * pow(zl, 3) +
                -0.09237745 * pow(zl, 4) + 0.03738027 * pow(zl, 5) + -0.005384159 * pow(zl, 6) +
                0.00042419 * pow(zl, 7);
  return beta;
}

template <typename T>
inline double get_harmonic_mean_denominator(T* M, uint32_t m) {
  double accumulator = 0.0;

  for (unsigned i = 0; i < m; i++) {
    accumulator += (1.0 / (1ULL << M[i]));
  }
  return accumulator;
}

template <typename T>
inline double get_beta_adjusted_estimate(const size_t m, const uint32_t z, T* M) {
  return (get_alpha(m) * m * (m - z) * (1 / (get_beta(z) + get_harmonic_mean_denominator(M, m))));
}

template <typename T>
inline double get_alpha_adjusted_estimate(const size_t m, T* M) {
  return (get_alpha(m) * m * m) * (1 / get_harmonic_mean_denominator(M, m));
};

template <typename T>
inline uint32_t count_zeros(T* M, size_t m) {
  uint32_t zeros = 0;
  for (uint32_t i = 0; i < m; i++) {
    if (M[i] == 0) {
      zeros++;
    }
  }
  return zeros;
}

template <class T>
inline size_t hll_size(const T* M, const size_t bitmap_sz_bits) {
  size_t m = 1 << bitmap_sz_bits;

  uint32_t zeros = count_zeros(M, m);
  double estimate = get_alpha_adjusted_estimate(m, M);
  if (estimate <= 2.5 * m) {
    if (zeros != 0) {
      estimate = m * log(static_cast<double>(m) / zeros);
    }
  } else {
    if (bitmap_sz_bits == 14) {  // Apply LogLog-Beta adjustment only when p=14
      estimate = get_beta_adjusted_estimate(m, zeros, M);
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
  // On the next line, 4 is the minimum for which we have an alpha adjustment factor in get_alpha()
  return std::min(16, std::max(static_cast<int>(k), 4));
}

extern int g_hll_precision_bits;

#endif  // QUERYENGINE_HYPERLOGLOG_H
