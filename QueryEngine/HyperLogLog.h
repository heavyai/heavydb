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
inline size_t hll_size(const T* M, const CountDistinctDescriptor& count_distinct_descriptor) {
  const double neg_pow_2_32 = -4294967296.0;
  const double pow_2_32 = 4294967296.0;
  size_t m = 1 << count_distinct_descriptor.bitmap_sz_bits;
  double sum{0};
  for (size_t i = 0; i < m; i++) {
    sum += 1.0 / (1 << M[i]);
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
  } else if (estimate > (1.0 / 30.0) * pow_2_32) {
    estimate = neg_pow_2_32 * log(1.0 - (estimate / pow_2_32));
  }
  return estimate;
}

template <class T>
inline void hll_unify(T* lhs, T* rhs, const size_t m) {
  for (size_t r = 0; r < m; ++r) {
    rhs[r] = lhs[r] = std::max(lhs[r], rhs[r]);
  }
}

const int HLL_MASK_WIDTH{11};

#endif  // QUERYENGINE_HYPERLOGLOG_H
