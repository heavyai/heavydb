/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <typeinfo>

#include "GfxDriver/RenderError.h"
#include "Shared/sqltypes.h"

namespace QueryRenderer {

template <typename T>
constexpr T getNullValue() {
  THROW_RUNTIME_EX("Values for nulls of type " + std::string(typeid(T).name()) +
                   " is not currently supported.");
  return T();
}

template <typename T>
T getNullValueFromTypeInfo(const SQLTypeInfo& type_info) {
  // TODO(croot): make specializations so that larger integral types cannot be converted
  // to smaller ones
  switch (type_info.get_type()) {
    case kBOOLEAN:
      return static_cast<T>(NULL_BOOLEAN);
    case kTINYINT:
      return static_cast<T>(NULL_TINYINT);
    case kSMALLINT:
      return static_cast<T>(NULL_SMALLINT);
    case kINT:
      return static_cast<T>(NULL_INT);
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      return static_cast<T>(NULL_BIGINT);
    default:
#ifndef __CUDACC__
      THROW_RUNTIME_EX("Converting integral null values from SQL type " +
                       type_info.get_type_name() + " is not currently supported.");
#else
      CHECK(false);
#endif  // not __CUDACC__
  }
  return T();
}

template <typename T>
inline bool isNullValue(const T val) {
  return val == getNullValue<T>();
}

template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
struct FloatingPtTypeSelector {
  using type = float;
};

/*
 * int specializations
 */
template <>
constexpr int getNullValue<int>() {
  return NULL_INT;
}

/*
 * uint specializations
 */
template <>
constexpr unsigned int getNullValue<unsigned int>() {
  // TODO(croot): have our own unsigned null values?
  return static_cast<unsigned int>(NULL_INT);
}

/*
 * float specializations
 */
template <>
constexpr float getNullValue<float>() {
  return NULL_FLOAT;
}

/*
 * double specializations
 */
template <>
constexpr double getNullValue<double>() {
  return NULL_DOUBLE;
}

template <>
struct FloatingPtTypeSelector<double> {
  using type = double;
};

/*
 * int64 specializations
 */
template <>
constexpr int64_t getNullValue<int64_t>() {
  return NULL_BIGINT;
}

template <>
struct FloatingPtTypeSelector<int64_t> {
  using type = double;
};

/*
 * uint64 specializations
 */
template <>
constexpr uint64_t getNullValue<uint64_t>() {
  // TODO(croot): have our own unsigned null values?
  return static_cast<uint64_t>(NULL_BIGINT);
}

template <>
struct FloatingPtTypeSelector<uint64_t> {
  using type = double;
};

/*
 * float specializations
 */
template <>
float getNullValueFromTypeInfo<float>(const SQLTypeInfo& type_info);

/*
 * double specializations
 */
template <>
double getNullValueFromTypeInfo<double>(const SQLTypeInfo& type_info);

}  // namespace QueryRenderer
