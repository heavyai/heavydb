/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Utils/NumericUtils.h"

namespace QueryRenderer {

namespace detail {

struct is_valid {
  const int64_t invalid_key;
  is_valid(const int64_t invalid_key) : invalid_key{invalid_key} {}
  __host__ __device__ bool operator()(const int64_t& val) { return val != invalid_key; }
};

// taken from SqlTypesLayout.h exp_to_scale
// TODO(croot): Copying that function here as otherwise #include<SqlTypesLayout.h>
// gets a bunch of compilation warnings that need to be fixed.
// Once those are fixed, we can remove this.
inline double exp_to_scale(const unsigned exp) {
  double res{1.0};
  for (unsigned i = 0; i < exp; ++i) {
    res *= 10.0;
  }
  return res;
}

struct DecimalConverter {
  const double scale;
  static constexpr int64_t decimal_null = getNullValue<int64_t>();
  static constexpr double double_null = getNullValue<double>();
  DecimalConverter(const unsigned exponent) : scale{exp_to_scale(exponent)} {}
  __host__ __device__ double operator()(const int64_t& val) const {
    if (val == decimal_null) {
      return double_null;
    }
    return static_cast<double>(val) / scale;
  }
};

}  // namespace detail

}  // namespace QueryRenderer
