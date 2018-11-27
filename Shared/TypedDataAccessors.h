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
#ifndef H_TypedDataAccessors__
#define H_TypedDataAccessors__

#include <math.h>
#include <string.h>
#include "Shared/sqltypes.h"

#ifndef CHECK  // if not collide with the one in glog/logging.h
#include "always_assert.h"
#endif

namespace {

template <typename LHT, typename RHT>
inline void value_truncated(const LHT& lhs, const RHT& rhs) {
  std::ostringstream os;
  os << "Value " << rhs << " would be truncated to "
     << (std::is_same<LHT, uint8_t>::value || std::is_same<LHT, int8_t>::value
             ? (int64_t)lhs
             : lhs);
  throw std::runtime_error(os.str());
};

template <typename T>
inline bool is_null(const T& v, const SQLTypeInfo& t) {
  if (std::is_floating_point<T>::value)
    return v == inline_fp_null_value<T>();
  switch (t.get_logical_size()) {
    case 1:
      return v == inline_int_null_value<int8_t>();
    case 2:
      return v == inline_int_null_value<int16_t>();
    case 4:
      return v == inline_int_null_value<int32_t>();
    case 8:
      return v == inline_int_null_value<int64_t>();
    default:
      abort();
  }
}

template <typename LHT, typename RHT>
inline bool integer_setter(LHT& lhs, const RHT& rhs, const SQLTypeInfo& t) {
  const int64_t r = is_null(rhs, t) ? inline_int_null_value<LHT>() : rhs;
  if ((lhs = r) != r)
    value_truncated(lhs, r);
  return true;
}

inline int get_element_size(const SQLTypeInfo& t) {
  if (t.is_string_array())
    return sizeof(int32_t);
  if (!t.is_array())
    return t.get_size();
  return SQLTypeInfo(t.get_subtype(),
                     t.get_dimension(),
                     t.get_scale(),
                     false,
                     t.get_compression(),
                     t.get_comp_param(),
                     kNULLT)
      .get_size();
}

inline bool is_null_string_index(const int size, const int32_t sidx) {
  switch (size) {
    case 1:
      return sidx == inline_int_null_value<uint8_t>();
    case 2:
      return sidx == inline_int_null_value<uint16_t>();
    case 4:
      return sidx == inline_int_null_value<int32_t>();
    default:
      abort();
  }
}

inline int32_t get_string_index(void* ptr, const int size) {
  switch (size) {
    case 1:
      return *(uint8_t*)ptr;
    case 2:
      return *(uint16_t*)ptr;
    case 4:
      return *(int32_t*)ptr;
    default:
      abort();
  }
}

inline bool set_string_index(void* ptr, const SQLTypeInfo& etype, int32_t sidx) {
  switch (get_element_size(etype)) {
    case 1:
      return integer_setter(*(uint8_t*)ptr, sidx, etype);
      break;
    case 2:
      return integer_setter(*(uint16_t*)ptr, sidx, etype);
      break;
    case 4:
      return integer_setter(*(int32_t*)ptr, sidx, etype);
      break;
    default:
      abort();
  }
}

template <typename T>
static void put_scalar(void* ndptr,
                       const SQLTypeInfo& etype,
                       const int esize,
                       const T oval) {
  // round floating oval to nearest integer
  auto rval = oval;
  if (std::is_floating_point<T>::value)
    if (etype.is_integer() || etype.is_time() || etype.is_timeinterval() ||
        etype.is_decimal())
      rval = round(rval);
  switch (etype.get_type()) {
    case kBOOLEAN:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kNUMERIC:
    case kDECIMAL:
      switch (esize) {
        case 1:
          integer_setter(*(int8_t*)ndptr, rval, etype);
          break;
        case 2:
          integer_setter(*(int16_t*)ndptr, rval, etype);
          break;
        case 4:
          integer_setter(*(int32_t*)ndptr, rval, etype);
          break;
        case 8:
          integer_setter(*(int64_t*)ndptr, rval, etype);
          break;
        default:
          abort();
      }
      break;
    case kFLOAT:
      *(float*)ndptr = rval;
      break;
    case kDOUBLE:
      *(double*)ndptr = rval;
      break;
    default:
      if (etype.is_string() && !etype.is_varlen())
        set_string_index(ndptr, etype, rval);
      else
        abort();
      break;
  }
}

inline double decimal_to_double(const SQLTypeInfo& otype, int64_t oval) {
  return oval / pow(10, otype.get_scale());
}

template <typename T>
inline void put_scalar(void* ndptr,
                       const SQLTypeInfo& ntype,
                       const T oval,
                       const std::string col_name,
                       const SQLTypeInfo* otype = nullptr) {
  const auto& etype = ntype.is_array() ? SQLTypeInfo(ntype.get_subtype(),
                                                     ntype.get_dimension(),
                                                     ntype.get_scale(),
                                                     ntype.get_notnull(),
                                                     ntype.get_compression(),
                                                     ntype.get_comp_param(),
                                                     kNULLT)
                                       : ntype;
  const auto esize = get_element_size(etype);
  const auto isnull = is_null(oval, etype);
  if (etype.get_notnull() && isnull)
    throw std::runtime_error("NULL value on NOT NULL column '" + col_name + "'");

  switch (etype.get_type()) {
    case kNUMERIC:
    case kDECIMAL:
      if (otype && otype->is_decimal())
        put_scalar<int64_t>(ndptr,
                            etype,
                            esize,
                            isnull ? inline_int_null_value<int64_t>()
                                   : convert_decimal_value_to_scale(oval, *otype, etype));
      else
        put_scalar<T>(ndptr,
                      etype,
                      esize,
                      isnull ? inline_int_null_value<int64_t>()
                             : oval * pow(10, etype.get_scale()));
      break;
    case kDATE:
      // For small dates, we store in days but decode in seconds
      // therefore we have to scale the decoded value in order to
      // make value storable.
      // Should be removed when we refactor code to use DateConverterFactory
      // from TargetValueConverterFactories so that we would
      // have everything in one place.
      if (etype.is_date_in_days()) {
        put_scalar<T>(ndptr,
                      etype,
                      get_element_size(etype),
                      isnull ? inline_int_null_value<int64_t>()
                             : static_cast<T>(oval / SECSPERDAY));
      } else {
        put_scalar<T>(ndptr, etype, get_element_size(etype), oval);
      }
      break;
    default:
      if (otype && otype->is_decimal())
        put_scalar<double>(ndptr, etype, decimal_to_double(*otype, oval), col_name);
      else
        put_scalar<T>(ndptr, etype, get_element_size(etype), oval);
      break;
  }
}

inline void put_null(void* ndptr, const SQLTypeInfo& ntype, const std::string col_name) {
  if (ntype.get_notnull())
    throw std::runtime_error("NULL value on NOT NULL column '" + col_name + "'");

  switch (ntype.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kNUMERIC:
    case kDECIMAL:
      switch (ntype.get_size()) {
        case 1:
          *(int8_t*)ndptr = inline_int_null_value<int8_t>();
          break;
        case 2:
          *(int16_t*)ndptr = inline_int_null_value<int16_t>();
          break;
        case 4:
          *(int32_t*)ndptr = inline_int_null_value<int32_t>();
          break;
        case 8:
          *(int64_t*)ndptr = inline_int_null_value<int64_t>();
          break;
        default:
          abort();
      }
      break;
    case kFLOAT:
      *(float*)ndptr = inline_fp_null_value<float>();
      break;
    case kDOUBLE:
      *(double*)ndptr = inline_fp_null_value<double>();
      break;
    default:
      //! this f is currently only for putting fixed-size data in place
      //! this f is not yet for putting var-size or dict-encoded data
      CHECK(false);
  }
}

template <typename T>
inline bool get_scalar(void* ndptr, const SQLTypeInfo& ntype, T& v) {
  switch (ntype.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kNUMERIC:
    case kDECIMAL:
      switch (ntype.get_size()) {
        case 1:
          return inline_int_null_value<int8_t>() == (v = *(int8_t*)ndptr);
        case 2:
          return inline_int_null_value<int16_t>() == (v = *(int16_t*)ndptr);
        case 4:
          return inline_int_null_value<int32_t>() == (v = *(int32_t*)ndptr);
        case 8:
          return inline_int_null_value<int64_t>() == (v = *(int64_t*)ndptr);
          break;
        default:
          abort();
      }
      break;
    case kFLOAT:
      return inline_fp_null_value<float>() == (v = *(float*)ndptr);
    case kDOUBLE:
      return inline_fp_null_value<double>() == (v = *(double*)ndptr);
    case kTEXT:
      v = get_string_index(ndptr, ntype.get_size());
      return is_null_string_index(ntype.get_size(), v);
    default:
      abort();
  }
}

template <typename T>
inline void set_minmax(T& min, T& max, const T val) {
  if (val < min)
    min = val;
  if (val > max)
    max = val;
}

}  // namespace
#endif
