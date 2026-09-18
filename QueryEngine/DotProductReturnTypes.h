/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file   DotProductReturnTypes.h
 * @brief  Shared lookup table of DOT_PRODUCT() return type based on its two input types.
 */
#pragma once

#include "Shared/sqltypes.h"

#include <array>

namespace heavyai {
namespace dot_product {

// Return type of DOT_PRODUCT() given both array types.
// Any changes to this table (that are not in the kDECIMAL row/column) must also
// be reflected in the BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT, ...)
// macros in QueryEngine/ArrayOps.cpp
// and class DotProduct in HeavyDBSqlOperatorTable.java.
// clang-format off
constexpr SQLTypes ret_types[]
   // kTINYINT kSMALLINT kINT     kBIGINT  kFLOAT   kDOUBLE  kDECIMAL
    { kINT,    kINT,     kBIGINT, kBIGINT, kFLOAT,  kDOUBLE, kFLOAT  // kTINYINT
    , kINT,    kINT,     kBIGINT, kBIGINT, kFLOAT,  kDOUBLE, kFLOAT  // kSMALLINT
    , kBIGINT, kBIGINT,  kBIGINT, kBIGINT, kDOUBLE, kDOUBLE, kDOUBLE // kINT
    , kBIGINT, kBIGINT,  kBIGINT, kBIGINT, kDOUBLE, kDOUBLE, kDOUBLE // kBIGINT
    , kFLOAT,  kFLOAT,   kDOUBLE, kDOUBLE, kFLOAT,  kDOUBLE, kFLOAT  // kFLOAT
    , kDOUBLE, kDOUBLE,  kDOUBLE, kDOUBLE, kDOUBLE, kDOUBLE, kDOUBLE // kDOUBLE
    , kFLOAT,  kFLOAT,   kDOUBLE, kDOUBLE, kFLOAT,  kDOUBLE, kDOUBLE // kDECIMAL
    };
// clang-format on
// These are the DOT_PRODUCT input types.
constexpr std::array<SQLTypes, 7u>
    types{kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE, kDECIMAL};
// Verify that the 7 x 7 table is the expected size.
static_assert(types.size() * types.size() == sizeof ret_types / sizeof *ret_types);

/// Verify that the ret_types array is symmetric.
constexpr bool isSymmetric(SQLTypes const* const arr, size_t const n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (arr[n * i + j] != arr[n * j + i]) {
        return false;
      }
    }
  }
  return true;
}
static_assert(isSymmetric(ret_types, types.size()));

}  // namespace dot_product
}  // namespace heavyai
