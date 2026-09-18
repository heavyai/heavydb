/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/NumericUtils.h"

namespace QueryRenderer {

template <>
float getNullValueFromTypeInfo<float>(const SQLTypeInfo& type_info) {
  switch (type_info.get_type()) {
    case kBOOLEAN:
      return static_cast<float>(NULL_BOOLEAN);
    case kTINYINT:
      return static_cast<float>(NULL_TINYINT);
    case kSMALLINT:
      return static_cast<float>(NULL_SMALLINT);
    case kINT:
      return static_cast<float>(NULL_INT);
    case kNUMERIC:
    case kDECIMAL:
    case kBIGINT:
      return static_cast<float>(NULL_BIGINT);
    case kFLOAT:
      return NULL_FLOAT;
    default:
#ifndef __CUDACC__
      THROW_RUNTIME_EX("Converting float null values from SQL type " +
                       type_info.get_type_name() + " is not currently supported.");
#else
      CHECK(false) << type_info.get_type();
#endif  // not __CUDACC__
  }
  return 0;
}

template <>
double getNullValueFromTypeInfo<double>(const SQLTypeInfo& type_info) {
  switch (type_info.get_type()) {
    case kBOOLEAN:
      return static_cast<double>(NULL_BOOLEAN);
    case kTINYINT:
      return static_cast<double>(NULL_TINYINT);
    case kSMALLINT:
      return static_cast<double>(NULL_SMALLINT);
    case kINT:
      return static_cast<double>(NULL_INT);
    case kNUMERIC:
    case kDECIMAL:
    case kBIGINT:
      return static_cast<double>(NULL_BIGINT);
    case kFLOAT:
      return static_cast<double>(NULL_FLOAT);
    case kDOUBLE:
      return NULL_DOUBLE;
    default:
#ifndef __CUDACC__
      THROW_RUNTIME_EX("Converting double null values from SQL type " +
                       type_info.get_type_name() + " is not currently supported.");
#else
      CHECK(false) << type_info.get_type();
#endif  // not __CUDACC__
  }
  return 0;
}

}  // namespace QueryRenderer
