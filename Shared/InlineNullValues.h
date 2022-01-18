/*
 * Copyright 2019 OmniSci, Inc.
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

#ifndef INLINENULLVALUES_H
#define INLINENULLVALUES_H

#include "../Logger/Logger.h"
#include "funcannotations.h"

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <limits>

#define NULL_BOOLEAN INT8_MIN
#define NULL_TINYINT INT8_MIN
#define NULL_SMALLINT INT16_MIN
#define NULL_INT INT32_MIN
#define NULL_BIGINT INT64_MIN
#define NULL_FLOAT FLT_MIN
#define NULL_DOUBLE DBL_MIN

#define NULL_ARRAY_BOOLEAN (INT8_MIN + 1)
#define NULL_ARRAY_TINYINT (INT8_MIN + 1)
#define NULL_ARRAY_SMALLINT (INT16_MIN + 1)
#define NULL_ARRAY_INT (INT32_MIN + 1)
#define NULL_ARRAY_BIGINT (INT64_MIN + 1)
#define NULL_ARRAY_FLOAT (FLT_MIN * 2.0)
#define NULL_ARRAY_DOUBLE (DBL_MIN * 2.0)

#define NULL_ARRAY_COMPRESSED_32 0x80000000U

#if !(defined(__CUDACC__) || defined(NO_BOOST))
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

template <class T>
constexpr inline int64_t inline_int_null_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::min()
                                  : std::numeric_limits<T>::max();
}

template <class T>
constexpr inline int64_t inline_int_null_array_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::min() + 1
                                  : std::numeric_limits<T>::max() - 1;
  // TODO: null_array values in signed types would step on max valid value
  // in fixlen unsigned arrays, the max valid value may need to be lowered.
}

template <class T>
constexpr inline int64_t max_valid_int_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::max()
                                  : std::numeric_limits<T>::max() - 1;
}

template <typename T>
constexpr inline T inline_fp_null_value() {
#if !(defined(__CUDACC__) || defined(NO_BOOST))
  LOG(FATAL) << "Only float or double overloads should be called.";
#else
  LOG(FATAL);
#endif
  return T{};
}

template <>
constexpr inline float inline_fp_null_value<float>() {
  return NULL_FLOAT;
}

template <>
constexpr inline double inline_fp_null_value<double>() {
  return NULL_DOUBLE;
}

template <typename T>
DEVICE T inline_fp_null_array_value() {
#if !(defined(__CUDACC__) || defined(NO_BOOST))
  LOG(FATAL) << "Only float or double overloads should be called.";
#else
  assert(false);
#endif
  return T{};
}

template <>
DEVICE inline float inline_fp_null_array_value<float>() {
  return NULL_ARRAY_FLOAT;
}

template <>
DEVICE inline double inline_fp_null_array_value<double>() {
  return NULL_ARRAY_DOUBLE;
}

#ifndef NO_BOOST
template <typename SQL_TYPE_INFO>
inline int64_t inline_int_null_val(const SQL_TYPE_INFO& ti) {
  auto type = ti.get_type();
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK_EQ(4, ti.get_logical_size());
    type = kINT;
  } else {
    CHECK_EQ(kENCODING_NONE, ti.get_compression());
  }
  switch (type) {
    case kBOOLEAN:
      return inline_int_null_value<int8_t>();
    case kTINYINT:
      return inline_int_null_value<int8_t>();
    case kSMALLINT:
      return inline_int_null_value<int16_t>();
    case kINT:
      return inline_int_null_value<int32_t>();
    case kBIGINT:
      return inline_int_null_value<int64_t>();
    case kTIMESTAMP:
    case kTIME:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return inline_int_null_value<int64_t>();
    case kDECIMAL:
    case kNUMERIC:
      return inline_int_null_value<int64_t>();
    default:
      abort();
  }
}

template <typename SQL_TYPE_INFO>
inline int64_t inline_fixed_encoding_null_val(const SQL_TYPE_INFO& ti) {
  if (ti.get_compression() == kENCODING_NONE) {
    return inline_int_null_val(ti);
  }
  if (ti.get_compression() == kENCODING_DATE_IN_DAYS) {
    switch (ti.get_comp_param()) {
      case 0:
      case 32:
        return inline_int_null_value<int32_t>();
      case 16:
        return inline_int_null_value<int16_t>();
      default:
#ifndef __CUDACC__
        CHECK(false) << "Unknown encoding width for date in days: "
                     << ti.get_comp_param();
#else
        CHECK(false);
#endif
    }
  }
  if (ti.get_compression() == kENCODING_DICT) {
    CHECK(ti.is_string());
    switch (ti.get_size()) {
      case 1:
        return inline_int_null_value<uint8_t>();
      case 2:
        return inline_int_null_value<uint16_t>();
      case 4:
        return inline_int_null_value<int32_t>();
      default:
#ifndef __CUDACC__
        CHECK(false) << "Unknown size for dictionary encoded type: " << ti.get_size();
#else
        CHECK(false);
#endif
    }
  }
  CHECK_EQ(kENCODING_FIXED, ti.get_compression());
  CHECK(ti.is_integer() || ti.is_time() || ti.is_decimal());
  CHECK_EQ(0, ti.get_comp_param() % 8);
  return -(1LL << (ti.get_comp_param() - 1));
}

template <typename SQL_TYPE_INFO>
inline double inline_fp_null_val(const SQL_TYPE_INFO& ti) {
  CHECK(ti.is_fp());
  const auto type = ti.get_type();
  switch (type) {
    case kFLOAT:
      return inline_fp_null_value<float>();
    case kDOUBLE:
      return inline_fp_null_value<double>();
    default:
      abort();
  }
}

// NULL_ARRAY sentinels
template <typename SQL_TYPE_INFO>
inline int64_t inline_int_null_array_val(const SQL_TYPE_INFO& ti) {
  auto type = ti.get_type();
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK_EQ(4, ti.get_logical_size());
    type = kINT;
  } else {
    CHECK_EQ(kENCODING_NONE, ti.get_compression());
  }
  // For all of the types below NULL sentinel is min of the range,
  // the value right above it is the NULL_ARRAY sentinel
  switch (type) {
    case kBOOLEAN:
      return inline_int_null_array_value<int8_t>();
    case kTINYINT:
      return inline_int_null_array_value<int8_t>();
    case kSMALLINT:
      return inline_int_null_array_value<int16_t>();
    case kINT:
      return inline_int_null_array_value<int32_t>();
    case kBIGINT:
      return inline_int_null_array_value<int64_t>();
    case kTIMESTAMP:
    case kTIME:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return inline_int_null_array_value<int64_t>();
    case kDECIMAL:
    case kNUMERIC:
      return inline_int_null_array_value<int64_t>();
    default:
      abort();
  }
}

template <typename SQL_TYPE_INFO>
inline int64_t inline_fixed_encoding_null_array_val(const SQL_TYPE_INFO& ti) {
  if (ti.get_compression() == kENCODING_NONE) {
    return inline_int_null_array_val(ti);
  }
  if (ti.get_compression() == kENCODING_DATE_IN_DAYS) {
    switch (ti.get_comp_param()) {
      case 0:
      case 32:
        return inline_int_null_array_value<int32_t>();
      case 16:
        return inline_int_null_array_value<int16_t>();
      default:
#ifndef __CUDACC__
        CHECK(false) << "Unknown encoding width for date in days: "
                     << ti.get_comp_param();
#else
        CHECK(false);
#endif
    }
  }
  if (ti.get_compression() == kENCODING_DICT) {
    CHECK(ti.is_string());
#ifndef __CUDACC__
    CHECK(false) << "Currently don't support fixed length arrays of dict encoded strings";
#else
    CHECK(false);
#endif
    switch (ti.get_size()) {
      case 1:
        return inline_int_null_array_value<uint8_t>();
      case 2:
        return inline_int_null_array_value<uint16_t>();
      case 4:
        return inline_int_null_array_value<int32_t>();
      default:
#ifndef __CUDACC__
        CHECK(false) << "Unknown size for dictionary encoded type: " << ti.get_size();
#else
        CHECK(false);
#endif
    }
  }
#ifndef __CUDACC__
  CHECK(false) << "Currently don't support fixed length arrays with fixed encoding";
#else
  CHECK(false);
#endif
  CHECK_EQ(kENCODING_FIXED, ti.get_compression());
  CHECK(ti.is_integer() || ti.is_time() || ti.is_decimal());
  CHECK_EQ(0, ti.get_comp_param() % 8);
  // The value of the NULL sentinel for fixed encoding is:
  //   -(1LL << (ti.get_comp_param() - 1))
  // NULL_ARRAY sentinel would have to be the value just above NULL:
  return -(1LL << (ti.get_comp_param() - 1)) + 1;
}

#endif  // NO_BOOST

#include <type_traits>

namespace serialize_detail {
template <int overload>
struct IntType;
template <>
struct IntType<1> {
  using type = uint8_t;
};
template <>
struct IntType<2> {
  using type = uint16_t;
};
template <>
struct IntType<4> {
  using type = uint32_t;
};
template <>
struct IntType<8> {
  using type = uint64_t;
};
}  // namespace serialize_detail

template <typename T, bool array = false>
CONSTEXPR DEVICE inline typename serialize_detail::IntType<sizeof(T)>::type
serialized_null_value() {
  using TT = typename serialize_detail::IntType<sizeof(T)>::type;
  T nv = 0;
  if
    CONSTEXPR(std::is_floating_point<T>::value) {
      if
        CONSTEXPR(array) { nv = inline_fp_null_array_value<T>(); }
      else {
        nv = inline_fp_null_value<T>();
      }
    }
  else if
    CONSTEXPR(std::is_integral<T>::value) {
      if
        CONSTEXPR(array) { nv = inline_int_null_array_value<T>(); }
      else {
        nv = inline_int_null_value<T>();
      }
    }
#if !(defined(__CUDACC__) || defined(NO_BOOST))
  else {
    CHECK(false) << "Serializing null values of floating point or integral types only is "
                    "supported.";
  }
#endif
  return *(TT*)(&nv);
}

template <typename T, bool array = false>
CONSTEXPR DEVICE inline bool is_null(const T& value) {
  using TT = typename serialize_detail::IntType<sizeof(T)>::type;
  return serialized_null_value<T, array>() == *(TT*)(&value);
}

template <typename T, bool array = false>
CONSTEXPR DEVICE inline void set_null(T& value) {
  using TT = typename serialize_detail::IntType<sizeof(T)>::type;
  *(TT*)(&value) = serialized_null_value<T, array>();
}

template <typename V,
          std::enable_if_t<!std::is_same<V, bool>::value && std::is_integral<V>::value,
                           int> = 0>
inline V inline_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V inline_null_value() {
  return inline_fp_null_value<V>();
}

#endif
