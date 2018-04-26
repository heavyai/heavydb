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
#include <math.h>
#include "sqltypes.h"
#include "always_assert.h"

static inline int get_uncompressed_element_size(const SQLTypeInfo& t) {
  if (t.is_string_array())
    return sizeof(int32_t);
  if (!t.is_array())
    return t.get_size();
  return SQLTypeInfo(t.get_subtype(), t.get_dimension(), t.get_scale(), false).get_size();
}

static inline uint32_t get_string_index(void* ptr, const int size) {
  switch (size) {
    case 1:
      return *(uint8_t*)ptr;
    case 2:
      return *(uint16_t*)ptr;
    case 4:
      return *(uint32_t*)ptr;
    default:
      CHECK(false);
      return 0;
  }
}

static inline bool set_string_index(void* ptr, const int size, const uint32_t rval) {
  switch (size) {
    case 1:
      return (*(uint8_t*)ptr = rval) == rval;
    case 2:
      return (*(uint16_t*)ptr = rval) == rval;
    case 4:
      return (*(uint32_t*)ptr = rval) == rval;
    default:
      CHECK(false);
      return false;
  }
}

static inline SQLTypes get_decimal_int_type(const SQLTypeInfo& ti) {
  switch (ti.get_size()) {
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

template <typename T>
static void put_scalar(void* ndptr, const SQLTypes etype, const int esize, const T oval) {
  // round floating oval to nearest integer
  auto rval = oval;
  if (std::is_floating_point<T>::value)
    if (IS_INTEGER(etype) || IS_TIME(etype))
      rval = round(rval);
  switch (etype) {
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      *(int64_t*)ndptr = rval;
      break;
    case kBOOLEAN:
      *(int8_t*)ndptr = 0 != rval;
      break;
    case kSMALLINT:
      *(int16_t*)ndptr = rval;
      break;
    case kINT:
      *(int32_t*)ndptr = rval;
      break;
    case kBIGINT:
      *(int64_t*)ndptr = rval;
      break;
    case kFLOAT:
      *(float*)ndptr = rval;
      break;
    case kDOUBLE:
      *(double*)ndptr = rval;
      break;
    default:
      if (IS_STRING(etype)) {
        if (!set_string_index(ndptr, esize, rval))
          throw std::runtime_error("Dictionary index " + std::to_string(rval) + " can't fit in " +
                                   std::to_string(esize) + " bytes.");
      } else
        CHECK(false);
      break;
  }
}

static inline double decimal_to_double(const SQLTypeInfo& otype, int64_t oval) {
  return oval / pow(10, otype.get_scale());
}

template <typename T>
static inline void put_scalar(void* ndptr, const SQLTypeInfo& ntype, const T oval, const SQLTypeInfo* otype = nullptr) {
  auto etype = ntype.is_array() ? ntype.get_subtype() : ntype.get_type();
  switch (etype) {
    case kNUMERIC:
    case kDECIMAL:
      if (otype && otype->is_decimal())
        put_scalar<int64_t>(ndptr, get_decimal_int_type(ntype), 0, convert_decimal_value_to_scale(oval, *otype, ntype));
      else
        put_scalar<T>(ndptr, get_decimal_int_type(ntype), 0, oval * pow(10, ntype.get_scale()));
      break;
    default:
      if (otype && otype->is_decimal())
        put_scalar<double>(ndptr, ntype, decimal_to_double(*otype, oval));
      else
        put_scalar<T>(ndptr, etype, get_uncompressed_element_size(ntype), oval);
      break;
  }
}

static inline void put_null(void* ndptr, const SQLTypeInfo& ntype, const std::string col_name) {
  if (ntype.get_notnull())
    throw std::runtime_error("NULL value on NOT NULL column '" + col_name + "'");

  switch (ntype.get_type()) {
    case kBOOLEAN:
      *(int8_t*)ndptr = NULL_BOOLEAN;
      break;
    case kSMALLINT:
      *(int16_t*)ndptr = NULL_SMALLINT;
      break;
    case kINT:
      *(int32_t*)ndptr = NULL_INT;
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      *(int64_t*)ndptr = NULL_BIGINT;
      break;
    case kFLOAT:
      *(float*)ndptr = NULL_FLOAT;
      break;
    case kDOUBLE:
      *(double*)ndptr = NULL_DOUBLE;
      break;
    default:
      //! this f is currently only for putting fixed-size data in place
      //! this f is not yet for putting var-size or dict-encoded data
      CHECK(false);
      break;
  }
}

template <typename T>
static inline bool get_scalar(void* ndptr, const SQLTypeInfo& ntype, T& v) {
  switch (ntype.get_type()) {
    case kBOOLEAN:
      return NULL_BOOLEAN == (v = *(int8_t*)ndptr);
    case kSMALLINT:
      return NULL_SMALLINT == (v = *(int16_t*)ndptr);
    case kINT:
      return NULL_INT == (v = *(int32_t*)ndptr);
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return NULL_BIGINT == (v = *(int64_t*)ndptr);
    case kNUMERIC:
    case kDECIMAL:
      switch (ntype.get_size()) {
        case 2:
          return NULL_SMALLINT == (v = *(int16_t*)ndptr);
        case 4:
          return NULL_INT == (v = *(int32_t*)ndptr);
        case 8:
          return NULL_BIGINT == (v = *(int64_t*)ndptr);
        default:
          CHECK(false);
      }
    case kFLOAT:
      return NULL_FLOAT == (v = *(float*)ndptr);
    case kDOUBLE:
      return NULL_DOUBLE == (v = *(double*)ndptr);
    case kTEXT:
      return NULL_BIGINT == (v = get_string_index(ndptr, get_uncompressed_element_size(ntype)));
    default:
      CHECK(false);
      return false;
  }
}

template <typename T>
static inline void set_minmax(T& min, T& max, const T val) {
  if (val < min)
    min = val;
  if (val > max)
    max = val;
}
