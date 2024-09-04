/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file		sqltypes.h
 * @brief		Constants for Builtin SQL Types supported by HEAVY.AI
 *
 */

#pragma once

#if !(defined(__CUDACC__) || defined(NO_BOOST))
#include "toString.h"
#else
#ifndef PRINT
#define PRINT(...)
#endif
#endif

#include "../Logger/Logger.h"
#include "../QueryEngine/Utils/FlatBuffer.h"
#include "Datum.h"
#include "funcannotations.h"
#include "sqltypes_lite.h"

#include <cassert>
#include <ctime>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "Shared/DbObjectKeys.h"

namespace sql_constants {
/*
The largest precision an SQL type is allowed to specify is currently 18 digits,
however, the most precise numeric value we can represent is actually precise to 19 digits.
This means that we can be slightly more relaxed when doing internal calculations than when
setting column types (e.g. a CAST from double to numeric could use precision 19 as long as
it doesn't overflow but a column cannot be specified to have precision 19+).
*/
constexpr static int32_t kMaxNumericPrecision =
    std::numeric_limits<int64_t>::digits10;  // 18
constexpr static int32_t kMaxRepresentableNumericPrecision =
    kMaxNumericPrecision + 1;  // 19
}  // namespace sql_constants

// must not change because these values persist in catalogs.
enum SQLTypes {
  kNULLT = 0,  // type for null values
  kBOOLEAN = 1,
  kCHAR = 2,
  kVARCHAR = 3,
  kNUMERIC = 4,
  kDECIMAL = 5,
  kINT = 6,
  kSMALLINT = 7,
  kFLOAT = 8,
  kDOUBLE = 9,
  kTIME = 10,
  kTIMESTAMP = 11,
  kBIGINT = 12,
  kTEXT = 13,
  kDATE = 14,
  kARRAY = 15,
  kINTERVAL_DAY_TIME = 16,
  kINTERVAL_YEAR_MONTH = 17,
  kPOINT = 18,
  kLINESTRING = 19,
  kPOLYGON = 20,
  kMULTIPOLYGON = 21,
  kTINYINT = 22,
  kGEOMETRY = 23,
  kGEOGRAPHY = 24,
  kEVAL_CONTEXT_TYPE = 25,  // Placeholder Type for ANY
  kVOID = 26,
  kCURSOR = 27,
  kCOLUMN = 28,
  kCOLUMN_LIST = 29,
  kMULTILINESTRING = 30,
  kMULTIPOINT = 31,
  kSQLTYPE_LAST = 32
};

#if !(defined(__CUDACC__) || defined(NO_BOOST))

inline std::string toString(const SQLTypes& type) {
  switch (type) {
    case kNULLT:
      return "NULL";
    case kBOOLEAN:
      return "BOOL";
    case kCHAR:
      return "CHAR";
    case kVARCHAR:
      return "VARCHAR";
    case kNUMERIC:
      return "NUMERIC";
    case kDECIMAL:
      return "DECIMAL";
    case kINT:
      return "INT";
    case kSMALLINT:
      return "SMALLINT";
    case kFLOAT:
      return "FLOAT";
    case kDOUBLE:
      return "DOUBLE";
    case kTIME:
      return "TIME";
    case kTIMESTAMP:
      return "TIMESTAMP";
    case kBIGINT:
      return "BIGINT";
    case kTEXT:
      return "TEXT";
    case kDATE:
      return "DATE";
    case kARRAY:
      return "ARRAY";
    case kINTERVAL_DAY_TIME:
      return "DAY TIME INTERVAL";
    case kINTERVAL_YEAR_MONTH:
      return "YEAR MONTH INTERVAL";
    case kPOINT:
      return "POINT";
    case kMULTIPOINT:
      return "MULTIPOINT";
    case kLINESTRING:
      return "LINESTRING";
    case kMULTILINESTRING:
      return "MULTILINESTRING";
    case kPOLYGON:
      return "POLYGON";
    case kMULTIPOLYGON:
      return "MULTIPOLYGON";
    case kTINYINT:
      return "TINYINT";
    case kGEOMETRY:
      return "GEOMETRY";
    case kGEOGRAPHY:
      return "GEOGRAPHY";
    case kEVAL_CONTEXT_TYPE:
      return "UNEVALUATED ANY";
    case kVOID:
      return "VOID";
    case kCURSOR:
      return "CURSOR";
    case kCOLUMN:
      return "COLUMN";
    case kCOLUMN_LIST:
      return "COLUMN_LIST";
    case kSQLTYPE_LAST:
      break;
  }
  LOG(FATAL) << "Invalid SQL type: " << type;
  return "";
}

inline std::ostream& operator<<(std::ostream& os, SQLTypes const sql_type) {
  os << toString(sql_type);
  return os;
}

#endif  // #if !(defined(__CUDACC__) || defined(NO_BOOST))

struct DoNothingDeleter {
  void operator()(int8_t*) {}
};
struct FreeDeleter {
  void operator()(int8_t* p) { free(p); }
};

struct HostArrayDatum : public VarlenDatum {
  using ManagedPtr = std::shared_ptr<int8_t>;

  HostArrayDatum() = default;

  HostArrayDatum(size_t const l, ManagedPtr p, bool const n)
      : VarlenDatum(l, p.get(), n), data_ptr(p) {}

  HostArrayDatum(size_t const l, int8_t* p, bool const n)
      : VarlenDatum(l, p, n), data_ptr(p, FreeDeleter()){};

  template <typename CUSTOM_DELETER,
            typename = std::enable_if_t<
                std::is_void<std::invoke_result_t<CUSTOM_DELETER, int8_t*> >::value> >
  HostArrayDatum(size_t const l, int8_t* p, CUSTOM_DELETER custom_deleter)
      : VarlenDatum(l, p, 0 == l), data_ptr(p, custom_deleter) {}

  template <typename CUSTOM_DELETER,
            typename = std::enable_if_t<
                std::is_void<std::invoke_result_t<CUSTOM_DELETER, int8_t*> >::value> >
  HostArrayDatum(size_t const l, int8_t* p, bool const n, CUSTOM_DELETER custom_deleter)
      : VarlenDatum(l, p, n), data_ptr(p, custom_deleter) {}

  ManagedPtr data_ptr;
};

struct DeviceArrayDatum : public VarlenDatum {
  DEVICE DeviceArrayDatum() : VarlenDatum() {}
};

inline DEVICE constexpr bool is_cuda_compiler() {
#ifdef __CUDACC__
  return true;
#else
  return false;
#endif
}

using ArrayDatum =
    std::conditional_t<is_cuda_compiler(), DeviceArrayDatum, HostArrayDatum>;

#ifndef __CUDACC__

struct DataBlockPtr {
  union {
    int8_t* numbersPtr;
    std::string* stringsPtr;
    ArrayDatum* arraysPtr;
  };

  size_t size;

  template <typename VectorType>
  void setStringsPtr(VectorType& vector) {
    static_assert(std::is_same<typename VectorType::value_type, std::string>::value,
                  "Only supported for string vectors");
    this->stringsPtr = vector.data();
    this->size = vector.size();
  }

  template <typename VectorType>
  void setArraysPtr(VectorType& vector) {
    static_assert(std::is_same<typename VectorType::value_type, ArrayDatum>::value,
                  "Only supported for array vectors");
    this->arraysPtr = vector.data();
    this->size = vector.size();
  }
};
#endif

// must not change because these values persist in catalogs.
enum EncodingType {
  kENCODING_NONE = 0,          // no encoding
  kENCODING_FIXED = 1,         // Fixed-bit encoding
  kENCODING_RL = 2,            // Run Length encoding
  kENCODING_DIFF = 3,          // Differential encoding
  kENCODING_DICT = 4,          // Dictionary encoding
  kENCODING_SPARSE = 5,        // Null encoding for sparse columns
  kENCODING_GEOINT = 6,        // Encoding coordinates as intergers
  kENCODING_DATE_IN_DAYS = 7,  // Date encoding in days
  kENCODING_ARRAY = 8,         // Array encoding for columns of arrays
  kENCODING_ARRAY_DICT = 9,    // Array encoding for columns of text encoding dict arrays
  kENCODING_LAST = 10
};

#if !(defined(__CUDACC__) || defined(NO_BOOST))

inline std::ostream& operator<<(std::ostream& os, EncodingType const type) {
  switch (type) {
    case kENCODING_NONE:
      os << "NONE";
      break;
    case kENCODING_FIXED:
      os << "FIXED";
      break;
    case kENCODING_RL:
      os << "RL";
      break;
    case kENCODING_DIFF:
      os << "DIFF";
      break;
    case kENCODING_DICT:
      os << "DICT";
      break;
    case kENCODING_SPARSE:
      os << "SPARSE";
      break;
    case kENCODING_GEOINT:
      os << "GEOINT";
      break;
    case kENCODING_DATE_IN_DAYS:
      os << "DATE_IN_DAYS";
      break;
    case kENCODING_ARRAY:
      os << "ARRAY";
      break;
    case kENCODING_ARRAY_DICT:
      os << "ARRAY_DICT";
      break;
    case kENCODING_LAST:
      break;
    default:
      LOG(FATAL) << "Invalid EncodingType: " << type;
  }
  return os;
}

inline std::string toString(const EncodingType& type) {
  std::ostringstream ss;
  ss << type;
  return ss.str();
}

#endif  // #if !(defined(__CUDACC__) || defined(NO_BOOST))

#define IS_INTEGER(T) \
  (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kBIGINT) || ((T) == kTINYINT))
#define IS_NUMBER(T)                                                             \
  (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kDOUBLE) || ((T) == kFLOAT) || \
   ((T) == kBIGINT) || ((T) == kNUMERIC) || ((T) == kDECIMAL) || ((T) == kTINYINT))
#define IS_STRING(T) (((T) == kTEXT) || ((T) == kVARCHAR) || ((T) == kCHAR))
#define IS_GEO(T)                                                          \
  (((T) == kPOINT) || ((T) == kLINESTRING) || ((T) == kMULTILINESTRING) || \
   ((T) == kMULTIPOINT) || ((T) == kPOLYGON) || ((T) == kMULTIPOLYGON))
#define IS_INTERVAL(T) ((T) == kINTERVAL_DAY_TIME || (T) == kINTERVAL_YEAR_MONTH)
#define IS_DECIMAL(T) ((T) == kNUMERIC || (T) == kDECIMAL)
#define IS_GEO_POLY(T) (((T) == kPOLYGON) || ((T) == kMULTIPOLYGON))
#define IS_GEO_LINE(T) (((T) == kLINESTRING) || ((T) == kMULTILINESTRING))
#define IS_GEO_MULTI(T) \
  (((T) == kMULTIPOLYGON) || ((T) == kMULTILINESTRING) || ((T) == kMULTIPOINT))

#include "InlineNullValues.h"

#define TRANSIENT_DICT(ID) (-(ID))
#define REGULAR_DICT(TRANSIENTID) (-(TRANSIENTID))

constexpr auto is_datetime(SQLTypes type) {
  return type == kTIME || type == kTIMESTAMP || type == kDATE;
}

// @type SQLTypeInfo
// @brief a structure to capture all type information including
// length, precision, scale, etc.
class SQLTypeInfo {
 public:
  SQLTypeInfo(SQLTypes t, int d, int s, bool n, EncodingType c, int p, SQLTypes st)
      : type(t)
      , subtype(st)
      , dimension(d)
      , scale(s)
      , notnull(n)
      , compression(c)
      , comp_param(p)
      , size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t, int d, int s, bool n)
      : type(t)
      , subtype(kNULLT)
      , dimension(d)
      , scale(s)
      , notnull(n)
      , compression(kENCODING_NONE)
      , comp_param(0)
      , size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t, EncodingType c, int p, SQLTypes st)
      : type(t)
      , subtype(st)
      , dimension(0)
      , scale(0)
      , notnull(false)
      , compression(c)
      , comp_param(p)
      , size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t, int d, int s) : SQLTypeInfo(t, d, s, false) {}
  SQLTypeInfo(SQLTypes t, bool n)
      : type(t)
      , subtype(kNULLT)
      , dimension(0)
      , scale(0)
      , notnull(n)
      , compression(kENCODING_NONE)
      , comp_param(0)
      , size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t) : SQLTypeInfo(t, false) {}
  SQLTypeInfo(SQLTypes t, bool n, EncodingType c)
      : type(t)
      , subtype(kNULLT)
      , dimension(0)
      , scale(0)
      , notnull(n)
      , compression(c)
      , comp_param(0)
      , size(get_storage_size()) {}
  SQLTypeInfo()
      : type(kNULLT)
      , subtype(kNULLT)
      , dimension(0)
      , scale(0)
      , notnull(false)
      , compression(kENCODING_NONE)
      , comp_param(0)
      , size(0) {}

  HOST DEVICE inline SQLTypes get_type() const { return type; }
  HOST DEVICE inline SQLTypes get_subtype() const { return subtype; }
  HOST DEVICE inline int get_dimension() const { return dimension; }
  inline int get_precision() const { return dimension; }
  HOST DEVICE inline int get_input_srid() const { return dimension; }
  HOST DEVICE inline int get_scale() const { return scale; }
  HOST DEVICE inline int get_output_srid() const { return scale; }
  HOST DEVICE inline bool get_notnull() const { return notnull; }
  HOST DEVICE inline EncodingType get_compression() const { return compression; }
  // TODO: Remove ambiguous `comp_param` attribute and replace with a comp_size enum.
  // dict_key should be used uniformly for dictionary ID.
  HOST DEVICE inline int get_comp_param() const { return comp_param; }
  HOST DEVICE inline int get_size() const { return size; }
  // Valid only for IS_STRING(T) types
  HOST DEVICE inline size_t get_max_strlen() const {
    return compression == kENCODING_DICT
               ? ~(~size_t(0) << 15)  // std::numeric_limits<int16_t>::max()
               : ~size_t(0);          // std::numeric_limits<size_t>::max()
  }

  inline int is_logical_geo_type() const {
    if (type == kPOINT || type == kLINESTRING || type == kMULTILINESTRING ||
        type == kMULTIPOINT || type == kPOLYGON || type == kMULTIPOLYGON) {
      return true;
    }
    return false;
  }

  inline bool is_variable_size() const { return size == -1; }

  inline int get_logical_size() const {
    if (compression == kENCODING_FIXED || compression == kENCODING_DATE_IN_DAYS) {
      SQLTypeInfo ti(type, dimension, scale, notnull, kENCODING_NONE, 0, subtype);
      return ti.get_size();
    }
    if (compression == kENCODING_DICT) {
      return 4;
    }
    return get_size();
  }

  inline int get_physical_cols() const {
    switch (type) {
      case kPOINT:
        return 1;  // coords
      case kMULTIPOINT:
        return 2;  // coords, bounds
      case kLINESTRING:
        return 2;  // coords, bounds
      case kMULTILINESTRING:
        return 3;  // coords, linestring_sizes, bounds
      case kPOLYGON:
        return 3;  // coords, ring_sizes, bounds
      case kMULTIPOLYGON:
        return 4;  // coords, ring_sizes, poly_rings, bounds
      default:
        break;
    }
    return 0;
  }
  inline int get_physical_coord_cols() const {
    // Return the number of extra columns which need to go through the executor,
    // as opposed to those which are only needed by CPU. In other words, we omit
    // any Bounds column.
    return has_bounds() ? get_physical_cols() - 1 : get_physical_cols();
  }
  inline bool has_bounds() const {
    switch (type) {
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        return true;
      default:
        break;
    }
    return false;
  }
  HOST DEVICE inline void set_type(SQLTypes t) { type = t; }
  HOST DEVICE inline void set_subtype(SQLTypes st) { subtype = st; }
  inline void set_dimension(int d) { dimension = d; }
  inline void set_precision(int d) { dimension = d; }
  inline void set_input_srid(int d) { dimension = d; }
  inline void set_scale(int s) { scale = s; }
  inline void set_output_srid(int s) { scale = s; }
  inline void set_notnull(bool n) { notnull = n; }
  inline void set_size(int s) { size = s; }
  inline void set_fixed_size() { size = get_storage_size(); }
  inline void set_dict_intersection() { dict_intersection = true; }
  inline void set_compression(EncodingType c) { compression = c; }
  inline void set_comp_param(int p) { comp_param = p; }
#ifndef __CUDACC__
  inline std::string get_type_name() const {
    if (IS_GEO(type)) {
      std::string srid_string = "";
      if (get_output_srid() > 0) {
        srid_string = ", " + std::to_string(get_output_srid());
      }
      CHECK_LT(static_cast<int>(subtype), kSQLTYPE_LAST);
      return type_name[static_cast<int>(subtype)] + "(" +
             type_name[static_cast<int>(type)] + srid_string + ")";
    }
    std::string ps = "";
    if (type == kDECIMAL || type == kNUMERIC) {
      ps = "(" + std::to_string(dimension) + "," + std::to_string(scale) + ")";
    } else if (type == kTIMESTAMP) {
      ps = "(" + std::to_string(dimension) + ")";
    }
    if (type == kARRAY) {
      auto elem_ti = get_elem_type();
      auto num_elems = (size > 0) ? std::to_string(size / elem_ti.get_size()) : "";
      CHECK_LT(static_cast<int>(subtype), kSQLTYPE_LAST);
      return elem_ti.get_type_name() + ps + "[" + num_elems + "]";
    }
    if (type == kCOLUMN) {
      auto elem_ti = get_elem_type();
      auto num_elems =
          (size > 0) ? "[" + std::to_string(size / elem_ti.get_size()) + "]" : "";
      CHECK_LT(static_cast<int>(subtype), kSQLTYPE_LAST);
      return "COLUMN<" + elem_ti.get_type_name() + ps + ">" + num_elems;
    }
    if (type == kCOLUMN_LIST) {
      auto elem_ti = get_elem_type();
      auto num_elems =
          (size > 0) ? "[" + std::to_string(size / elem_ti.get_size()) + "]" : "";
      CHECK_LT(static_cast<int>(subtype), kSQLTYPE_LAST);
      return "COLUMN_LIST<" + elem_ti.get_type_name() + ps + ">" + num_elems;
    }
    return type_name[static_cast<int>(type)] + ps;
  }
  inline std::string get_compression_name() const {
    return comp_name[(int)compression];
  }
  std::string toString() const {
    return to_string();
  }  // for PRINT macro
  inline std::string to_string() const {
    std::ostringstream oss;
    oss << "(type=" << type_name[static_cast<int>(type)]
        << ", dimension=" << get_dimension() << ", scale=" << get_scale()
        << ", null=" << (get_notnull() ? "not nullable" : "nullable")
        << ", compression_name=" << get_compression_name()
        << ", comp_param=" << get_comp_param()
        << ", subtype=" << type_name[static_cast<int>(subtype)] << ", size=" << get_size()
        << ", element_size=" << get_elem_type().get_size() << ", dict_key=" << dict_key_
        << ", uses_flatbuffer=" << uses_flatbuffer_ << ")";
    return oss.str();
  }

  inline std::string get_buffer_name() const {
    if (is_array()) {
      return "Array";
    }
    if (is_text_encoding_none()) {
      return "TextEncodingNone";
    }

    if (is_column()) {
      return "Column";
    }

    assert(false);
    return "";
  }
#endif
  template <SQLTypes... types>
  bool is_any() const {
    return (... || (types == type));
  }
  inline bool is_string() const {
    return IS_STRING(type);
  }
  inline bool is_string_array() const {
    return (type == kARRAY) && IS_STRING(subtype);
  }
  inline bool is_integer() const {
    return IS_INTEGER(type);
  }
  inline bool is_decimal() const {
    return type == kDECIMAL || type == kNUMERIC;
  }
  inline bool is_fp() const {
    return type == kFLOAT || type == kDOUBLE;
  }
  inline bool is_number() const {
    return IS_NUMBER(type);
  }
  inline bool is_time() const {
    return is_datetime(type);
  }
  inline bool is_boolean() const {
    return type == kBOOLEAN;
  }
  inline bool is_array() const {
    return type == kARRAY;
  }  // Array
  inline bool is_varlen_array() const {
    return type == kARRAY && size <= 0;
  }
  inline bool is_fixlen_array() const {
    return type == kARRAY && size > 0;
  }
  inline bool is_timeinterval() const {
    return IS_INTERVAL(type);
  }
  inline bool is_geometry() const {
    return IS_GEO(type);
  }
  inline bool is_column() const {
    return type == kCOLUMN;
  }  // Column
  inline bool is_column_list() const {
    return type == kCOLUMN_LIST;
  }  // ColumnList
  inline bool is_column_array() const {
    const auto c = get_compression();
    return type == kCOLUMN && (c == kENCODING_ARRAY || c == kENCODING_ARRAY_DICT);
  }  // ColumnArray
  inline bool is_column_list_array() const {
    const auto c = get_compression();
    return type == kCOLUMN_LIST && (c == kENCODING_ARRAY || c == kENCODING_ARRAY_DICT);
  }  // ColumnList of ColumnArray
  inline bool is_text_encoding_none() const {
    return type == kTEXT && get_compression() == kENCODING_NONE;
  }
  inline bool is_text_encoding_dict() const {
    return type == kTEXT && get_compression() == kENCODING_DICT;
  }
  inline bool is_text_encoding_dict_array() const {
    return type == kARRAY && subtype == kTEXT && get_compression() == kENCODING_DICT;
  }
  inline bool is_buffer() const {
    return is_array() || is_column() || is_column_list() || is_text_encoding_none();
  }
  inline bool transforms() const {
    return IS_GEO(type) && get_input_srid() > 0 && get_output_srid() > 0 &&
           get_output_srid() != get_input_srid();
  }

  inline bool is_varlen() const {  // TODO: logically this should ignore fixlen arrays
    return (IS_STRING(type) && compression != kENCODING_DICT) || type == kARRAY ||
           IS_GEO(type);
  }

  // need this here till is_varlen can be fixed w/o negative impact to existing code
  inline bool is_varlen_indeed() const {
    // SQLTypeInfo.is_varlen() is broken with fixedlen array now
    // and seems left broken for some concern, so fix it locally
    return is_varlen() && !is_fixlen_array();
  }

  inline bool is_dict_encoded_string() const {
    return is_string() && compression == kENCODING_DICT;
  }

  inline bool is_none_encoded_string() const {
    return is_string() && compression == kENCODING_NONE;
  }

  inline bool is_subtype_dict_encoded_string() const {
    return IS_STRING(subtype) && compression == kENCODING_DICT;
  }

  inline bool is_dict_encoded_type() const {
    return is_dict_encoded_string() ||
           (is_array() && get_elem_type().is_dict_encoded_string());
  }

  inline bool is_dict_intersection() const {
    return dict_intersection;
  }

  inline bool has_same_itemtype(const SQLTypeInfo& other) const {
    if ((is_column() || is_column_list()) &&
        (other.is_column() || other.is_column_list())) {
      return subtype == other.get_subtype() &&
             ((compression != kENCODING_ARRAY && compression != kENCODING_ARRAY_DICT) ||
              compression == other.get_compression());
    }
    return subtype == other.get_subtype();
  }

  inline bool is_extractable_int_type() const {
    // Usable with extract_int_type_from_datum().
    return (type == kBOOLEAN || is_integer() || is_dict_encoded_string() || is_time());
  }

  HOST DEVICE inline bool operator!=(const SQLTypeInfo& rhs) const {
    return type != rhs.get_type() || subtype != rhs.get_subtype() ||
           dimension != rhs.get_dimension() || scale != rhs.get_scale() ||
           compression != rhs.get_compression() ||
           (compression != kENCODING_NONE && comp_param != rhs.get_comp_param() &&
            comp_param != TRANSIENT_DICT(rhs.get_comp_param())) ||
           notnull != rhs.get_notnull() || dict_key_ != rhs.dict_key_;
  }
  HOST DEVICE inline bool operator==(const SQLTypeInfo& rhs) const {
    return type == rhs.get_type() && subtype == rhs.get_subtype() &&
           dimension == rhs.get_dimension() && scale == rhs.get_scale() &&
           compression == rhs.get_compression() &&
           (compression == kENCODING_NONE || comp_param == rhs.get_comp_param() ||
            comp_param == TRANSIENT_DICT(rhs.get_comp_param())) &&
           notnull == rhs.get_notnull() && dict_key_ == rhs.dict_key_;
  }

  inline int get_array_context_logical_size() const {
    if (is_string()) {
      auto comp_type(get_compression());
      if (comp_type == kENCODING_DICT || comp_type == kENCODING_FIXED ||
          comp_type == kENCODING_NONE) {
        return sizeof(int32_t);
      }
    }
    return get_logical_size();
  }

  HOST DEVICE inline void operator=(const SQLTypeInfo& rhs) {
    type = rhs.get_type();
    subtype = rhs.get_subtype();
    dimension = rhs.get_dimension();
    scale = rhs.get_scale();
    notnull = rhs.get_notnull();
    compression = rhs.get_compression();
    comp_param = rhs.get_comp_param();
    size = rhs.get_size();
    dict_key_ = rhs.dict_key_;
    uses_flatbuffer_ = rhs.uses_flatbuffer_;
  }

  inline bool is_castable(const SQLTypeInfo& new_type_info) const {
    // can always cast between the same type but different precision/scale/encodings
    if (type == new_type_info.get_type()) {
      return true;
      // can always cast between strings
    } else if (is_string() && new_type_info.is_string()) {
      return true;
    } else if (is_string() && !new_type_info.is_string()) {
      return false;
    } else if (!is_string() && new_type_info.is_string()) {
      return true;
      // can cast between numbers
    } else if (is_number() && new_type_info.is_number()) {
      return true;
      // can cast from timestamp or date to number (epoch)
    } else if ((type == kTIMESTAMP || type == kDATE) && new_type_info.is_number()) {
      return true;
      // can cast from number (epoch) to timestamp, date, or time
    } else if (is_number() && new_type_info.is_time()) {
      return true;
      // can cast from date to timestamp
    } else if (type == kDATE && new_type_info.get_type() == kTIMESTAMP) {
      return true;
    } else if (type == kTIMESTAMP && new_type_info.get_type() == kDATE) {
      return true;
    } else if (type == kTIMESTAMP && new_type_info.get_type() == kTIME) {
      return true;
    } else if (type == kBOOLEAN && new_type_info.is_number()) {
      return true;
    } else if (type == kARRAY && new_type_info.get_type() == kARRAY) {
      return get_elem_type().is_castable(new_type_info.get_elem_type());
    } else if (type == kCOLUMN && new_type_info.get_type() == kCOLUMN) {
      return get_elem_type().is_castable(new_type_info.get_elem_type());
    } else if (type == kCOLUMN_LIST && new_type_info.get_type() == kCOLUMN_LIST) {
      return get_elem_type().is_castable(new_type_info.get_elem_type());
    } else {
      return false;
    }
  }

  /**
   * @brief returns true if the sql_type can be cast to the type specified by
   * new_type_info with no loss of precision. Currently only used in
   * ExtensionFunctionsBindings to determine legal function matches, but we should
   * consider whether we need to rationalize implicit casting behavior more broadly in
   * QueryEngine.
   */

  inline bool is_numeric_scalar_auto_castable(const SQLTypeInfo& new_type_info) const {
    const auto& new_type = new_type_info.get_type();
    switch (type) {
      case kBOOLEAN:
        return new_type == kBOOLEAN;
      case kTINYINT:
      case kSMALLINT:
      case kINT:
        if (!new_type_info.is_number()) {
          return false;
        }
        if (new_type_info.is_fp()) {
          // We can lose precision here, but preserving existing behavior
          return true;
        }
        return new_type_info.get_logical_size() >= get_logical_size();
      case kBIGINT:
        return new_type == kBIGINT || new_type == kDOUBLE || new_type == kFLOAT;
      case kFLOAT:
      case kDOUBLE:
        if (!new_type_info.is_fp()) {
          return false;
        }
        return (new_type_info.get_logical_size() >= get_logical_size());
      case kDECIMAL:
      case kNUMERIC:
        switch (new_type) {
          case kDECIMAL:
          case kNUMERIC:
            return new_type_info.get_dimension() >= get_dimension();
          case kDOUBLE:
            return true;
          case kFLOAT:
            return get_dimension() <= 7;
          default:
            return false;
        }
      case kTIMESTAMP:
        if (new_type != kTIMESTAMP) {
          return false;
        }
        return new_type_info.get_dimension() >= get_dimension();
      case kDATE:
        return new_type == kDATE;
      case kTIME:
        return new_type == kTIME;
      default:
        UNREACHABLE();
        return false;
    }
  }

  /**
   * @brief returns integer between 1 and 8 indicating what is roughly equivalent to the
   * logical byte size of a scalar numeric type (including boolean + time types), but with
   * decimals and numerics mapped to the byte size of their dimension (which may vary from
   * the type width), and timestamps, dates and times handled in a relative fashion.
   * Note: this function only takes the scalar numeric types above, and will throw a check
   * for other types.
   */

  inline int32_t get_numeric_scalar_scale() const {
    CHECK(type == kBOOLEAN || type == kTINYINT || type == kSMALLINT || type == kINT ||
          type == kBIGINT || type == kFLOAT || type == kDOUBLE || type == kDECIMAL ||
          type == kNUMERIC || type == kTIMESTAMP || type == kDATE || type == kTIME);
    switch (type) {
      case kBOOLEAN:
        return 1;
      case kTINYINT:
      case kSMALLINT:
      case kINT:
      case kBIGINT:
      case kFLOAT:
      case kDOUBLE:
        return get_logical_size();
      case kDECIMAL:
      case kNUMERIC:
        if (get_dimension() > 7) {
          return 8;
        } else {
          return 4;
        }
      case kTIMESTAMP:
        switch (get_dimension()) {
          case 9:
            return 8;
          case 6:
            return 4;
          case 3:
            return 2;
          case 0:
            return 1;
          default:
            UNREACHABLE();
        }
      case kDATE:
        return 1;
      case kTIME:
        return 1;
      default:
        UNREACHABLE();
        return 0;
    }
  }

  HOST DEVICE inline bool is_null(const Datum& d) const {
    // assuming Datum is always uncompressed
    switch (type) {
      case kBOOLEAN:
        return (int8_t)d.boolval == NULL_BOOLEAN;
      case kTINYINT:
        return d.tinyintval == NULL_TINYINT;
      case kSMALLINT:
        return d.smallintval == NULL_SMALLINT;
      case kINT:
        return d.intval == NULL_INT;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        return d.bigintval == NULL_BIGINT;
      case kFLOAT:
        return d.floatval == NULL_FLOAT;
      case kDOUBLE:
        return d.doubleval == NULL_DOUBLE;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        return d.bigintval == NULL_BIGINT;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
        // @TODO handle null strings
        break;
      case kNULLT:
        return true;
      case kARRAY:
        return d.arrayval == NULL || d.arrayval->is_null;
      default:
        break;
    }
    return false;
  }
  HOST DEVICE inline bool is_null(const int8_t* val) const {
    if (type == kFLOAT) {
      return *(float*)val == NULL_FLOAT;
    }
    if (type == kDOUBLE) {
      return *(double*)val == NULL_DOUBLE;
    }
    // val can be either compressed or uncompressed
    switch (size) {
      case 1:
        return *val == NULL_TINYINT;
      case 2:
        return *(int16_t*)val == NULL_SMALLINT;
      case 4:
        return *(int32_t*)val == NULL_INT;
      case 8:
        return *(int64_t*)val == NULL_BIGINT;
      case kNULLT:
        return true;
      default:
        // @TODO(wei) handle null strings
        break;
    }
    return false;
  }
  HOST DEVICE inline bool is_null_fixlen_array(const int8_t* val, int array_size) const {
    // Check if fixed length array has a NULL_ARRAY sentinel as the first element
    if (type == kARRAY && val && array_size > 0 && array_size == size) {
      // Need to create element type to get the size, but can't call get_elem_type()
      // since this is a HOST DEVICE function. Going through copy constructor instead.
      auto elem_ti{*this};
      elem_ti.set_type(subtype);
      elem_ti.set_subtype(kNULLT);
      auto elem_size = elem_ti.get_storage_size();
      if (elem_size < 1) {
        return false;
      }
      if (subtype == kFLOAT) {
        return *(float*)val == NULL_ARRAY_FLOAT;
      }
      if (subtype == kDOUBLE) {
        return *(double*)val == NULL_ARRAY_DOUBLE;
      }
      switch (elem_size) {
        case 1:
          return *val == NULL_ARRAY_TINYINT;
        case 2:
          return *(int16_t*)val == NULL_ARRAY_SMALLINT;
        case 4:
          return *(int32_t*)val == NULL_ARRAY_INT;
        case 8:
          return *(int64_t*)val == NULL_ARRAY_BIGINT;
        default:
          return false;
      }
    }
    return false;
  }
  HOST DEVICE inline bool is_null_point_coord_array(const int8_t* val,
                                                    int array_size) const {
    if (type == kARRAY && subtype == kTINYINT && val && array_size > 0 &&
        array_size == size) {
      if (array_size == 2 * sizeof(double)) {
        return *(double*)val == NULL_ARRAY_DOUBLE;
      }
      if (array_size == 2 * sizeof(int32_t)) {
        return *(uint32_t*)val == NULL_ARRAY_COMPRESSED_32;
      }
    }
    return false;
  }

  inline SQLTypeInfo get_elem_type() const {
    SQLTypeInfo type_info = *this;
    if ((type == kCOLUMN || type == kCOLUMN_LIST) && compression == kENCODING_ARRAY) {
      type_info.set_type(kARRAY);
      type_info.set_compression(kENCODING_NONE);
    } else if ((type == kCOLUMN || type == kCOLUMN_LIST) &&
               compression == kENCODING_ARRAY_DICT) {
      type_info.set_type(kARRAY);
      type_info.set_compression(kENCODING_DICT);
    } else if ((type == kCOLUMN || type == kCOLUMN_LIST) && IS_GEO(subtype)) {
      type_info.set_type(subtype);
      type_info.set_subtype(kGEOMETRY);
    } else if (IS_GEO(type)) {
      if (type_info.get_compression() == kENCODING_GEOINT) {
        type_info.set_type(kINT);
        type_info.set_compression(kENCODING_NONE);
      } else {
        type_info.set_type(kDOUBLE);
      }
      type_info.set_subtype(kNULLT);
    } else if (type == kARRAY) {
      type_info.set_type(subtype);
      type_info.set_subtype(kNULLT);
      type_info.set_notnull(false);
      type_info.setUsesFlatBuffer(false);
    } else {
      type_info.set_type(subtype);
      type_info.set_subtype(kNULLT);
    }
    type_info.setStorageSize();
    return type_info;
  }

  inline SQLTypeInfo get_array_type() const {
    SQLTypeInfo type_info = *this;
    type_info.set_type(kARRAY);
    type_info.set_subtype(type);
    type_info.setStorageSize();
    return type_info;
  }

  inline bool is_date_in_days() const {
    if (type == kDATE) {
      const auto comp_type = get_compression();
      if (comp_type == kENCODING_DATE_IN_DAYS) {
        return true;
      }
    }
    return false;
  }

  inline bool is_date() const {
    return type == kDATE;
  }

  inline bool is_time_or_date() const {
    return type == kDATE || type == kTIME || type == kTIMESTAMP;
  }

  inline bool is_high_precision_timestamp() const {
    if (type == kTIMESTAMP) {
      const auto dimension = get_dimension();
      if (dimension > 0) {
        return true;
      }
    }
    return false;
  }

  inline bool is_timestamp() const {
    return type == kTIMESTAMP;
  }
  inline bool is_encoded_timestamp() const {
    return is_timestamp() && compression == kENCODING_FIXED;
  }

  void setStorageSize() {
    size = get_storage_size();
  }

  const shared::StringDictKey& getStringDictKey() const {
    // If comp_param is set, it should equal dict_id.
    CHECK(dict_key_.dict_id == comp_param || comp_param == 0);
    return dict_key_;
  }

  void setStringDictKey(const shared::StringDictKey& dict_key) {
    dict_key_ = dict_key;
    // If comp_param is set, it should equal dict_id.
    CHECK(dict_key_.dict_id == comp_param || comp_param == 0);
  }

  // TODO: Remove below *SkipCompParamCheck methods as part of comp_param refactor
  const shared::StringDictKey& getStringDictKeySkipCompParamCheck() const {
    return dict_key_;
  }

  void setStringDictKeySkipCompParamCheck(const shared::StringDictKey& dict_key) {
    dict_key_ = dict_key;
  }

  // a column or an ouput column of this type may use FlatBuffer storage.
  inline void setUsesFlatBuffer(bool uses_flatbuffer = true) {
    uses_flatbuffer_ = uses_flatbuffer;
  }

  inline bool usesFlatBuffer() const {
    return uses_flatbuffer_;
  }

  // Checks if a column of this type can use FlatBuffer storage
  inline bool supportsFlatBuffer() const {
    switch (type) {
      case kARRAY:
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOINT:
      case kMULTILINESTRING:
      case kMULTIPOLYGON:
        return true;
      case kTEXT:
        return get_compression() == kENCODING_NONE;
      case kCOLUMN:
      case kCOLUMN_LIST:
        // calling supportsFlatBuffer on a column type is not meaningful
        UNREACHABLE();
      default:;
    }
    return false;
  }

  SQLTypeInfoLite toLite() const {
    SQLTypeInfoLite ti_lite;
    switch (type) {
      case kPOINT:
        ti_lite.type = SQLTypeInfoLite::POINT;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kLINESTRING:
        ti_lite.type = SQLTypeInfoLite::LINESTRING;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kPOLYGON:
        ti_lite.type = SQLTypeInfoLite::POLYGON;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kMULTIPOINT:
        ti_lite.type = SQLTypeInfoLite::MULTIPOINT;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kMULTILINESTRING:
        ti_lite.type = SQLTypeInfoLite::MULTILINESTRING;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kMULTIPOLYGON:
        ti_lite.type = SQLTypeInfoLite::MULTIPOLYGON;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kTEXT:
        ti_lite.type = SQLTypeInfoLite::TEXT;
        ti_lite.subtype = SQLTypeInfoLite::UNSPECIFIED;
        break;
      case kARRAY:
        ti_lite.type = SQLTypeInfoLite::ARRAY;
        switch (subtype) {
          case kBOOLEAN:
            ti_lite.subtype = SQLTypeInfoLite::BOOLEAN;
            break;
          case kTINYINT:
            ti_lite.subtype = SQLTypeInfoLite::TINYINT;
            break;
          case kSMALLINT:
            ti_lite.subtype = SQLTypeInfoLite::SMALLINT;
            break;
          case kINT:
            ti_lite.subtype = SQLTypeInfoLite::INT;
            break;
          case kBIGINT:
            ti_lite.subtype = SQLTypeInfoLite::BIGINT;
            break;
          case kFLOAT:
            ti_lite.subtype = SQLTypeInfoLite::FLOAT;
            break;
          case kDOUBLE:
            ti_lite.subtype = SQLTypeInfoLite::DOUBLE;
            break;
          case kTEXT:
            ti_lite.subtype = SQLTypeInfoLite::TEXT;
            break;
          default:
            UNREACHABLE();
        }
        break;
      default:
        UNREACHABLE();
    }
    if (is_geometry()) {
      switch (get_compression()) {
        case kENCODING_NONE:
          ti_lite.compression = SQLTypeInfoLite::NONE;
          break;
        case kENCODING_GEOINT:
          ti_lite.compression = SQLTypeInfoLite::GEOINT;
          break;
        default:
          UNREACHABLE();
      }
      ti_lite.dimension = get_input_srid();
      ti_lite.scale = get_output_srid();
      ti_lite.db_id = 0;    // unused
      ti_lite.dict_id = 0;  // unused
    } else if (type == kTEXT) {
      switch (get_compression()) {
        case kENCODING_NONE:
          ti_lite.compression = SQLTypeInfoLite::NONE;
          break;
        case kENCODING_DICT:
          ti_lite.compression = SQLTypeInfoLite::DICT;
          break;
        default:
          UNREACHABLE();
      }
      ti_lite.dimension = 0;  // unused
      ti_lite.scale = 0;      // unused
      ti_lite.db_id = dict_key_.db_id;
      ti_lite.dict_id = dict_key_.dict_id;
    } else if (type == kARRAY) {
      if (subtype == kTEXT) {
        switch (get_compression()) {
          case kENCODING_NONE:
            ti_lite.compression = SQLTypeInfoLite::NONE;
            break;
          case kENCODING_DICT:
            ti_lite.compression = SQLTypeInfoLite::DICT;
            break;
          default:
            UNREACHABLE();
        }
      }
      ti_lite.dimension = 0;  // unused
      ti_lite.scale = 0;      // unused
      ti_lite.db_id = dict_key_.db_id;
      ti_lite.dict_id = dict_key_.dict_id;
    } else {
      UNREACHABLE();
    }
    return ti_lite;
  }

  FlatBufferManager::ValueType toValueType() const {
    SQLTypes sql_value_type =
        ((type == kCOLUMN || type == kCOLUMN_LIST || type == kARRAY) ? subtype : type);
    switch (sql_value_type) {
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOINT:
      case kMULTILINESTRING:
      case kMULTIPOLYGON:
        return (get_compression() == kENCODING_GEOINT ? FlatBufferManager::PointInt32
                                                      : FlatBufferManager::PointFloat64);
      case kBOOLEAN:
        return FlatBufferManager::Bool8;
      case kTINYINT:
        return FlatBufferManager::Int8;
      case kSMALLINT:
        return FlatBufferManager::Int16;
      case kINT:
        return FlatBufferManager::Int32;
      case kBIGINT:
        return FlatBufferManager::Int64;
      case kFLOAT:
        return FlatBufferManager::Float32;
      case kDOUBLE:
        return FlatBufferManager::Float64;
      case kTEXT: {
        switch (get_compression()) {
          case kENCODING_NONE:
            return FlatBufferManager::Int8;
          case kENCODING_DICT:
            return FlatBufferManager::Int32;
          default:
            UNREACHABLE();
        }
      } break;
      default:
        UNREACHABLE();
    }
    return {};
  }

 private:
  SQLTypes type;     // type id
  SQLTypes subtype;  // element type of arrays or columns
  int dimension;     // VARCHAR/CHAR length or NUMERIC/DECIMAL precision or COLUMN_LIST
                     // length or TIMESTAMP precision
  int scale;         // NUMERIC/DECIMAL scale
  bool notnull;      // nullable?  a hint, not used for type checking
  EncodingType compression;  // compression scheme
  int comp_param;            // compression parameter when applicable for certain schemes
  int size;                  // size of the type in bytes.  -1 for variable size
  bool dict_intersection{false};
#ifndef __CUDACC__
  static std::string type_name[kSQLTYPE_LAST];
  static std::string comp_name[kENCODING_LAST];
#endif
  shared::StringDictKey dict_key_;
  bool uses_flatbuffer_{false};
  HOST DEVICE inline int get_storage_size() const {
    switch (type) {
      case kBOOLEAN:
        return sizeof(int8_t);
      case kTINYINT:
        return sizeof(int8_t);
      case kSMALLINT:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(int16_t);
          case kENCODING_FIXED:
          case kENCODING_SPARSE:
            return comp_param / 8;
          case kENCODING_RL:
          case kENCODING_DIFF:
            break;
          default:
            assert(false);
        }
        break;
      case kINT:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(int32_t);
          case kENCODING_FIXED:
          case kENCODING_SPARSE:
          case kENCODING_GEOINT:
            return comp_param / 8;
          case kENCODING_RL:
          case kENCODING_DIFF:
            break;
          default:
            assert(false);
        }
        break;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(int64_t);
          case kENCODING_FIXED:
          case kENCODING_SPARSE:
            return comp_param / 8;
          case kENCODING_RL:
          case kENCODING_DIFF:
            break;
          default:
            assert(false);
        }
        break;
      case kFLOAT:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(float);
          case kENCODING_FIXED:
          case kENCODING_RL:
          case kENCODING_DIFF:
          case kENCODING_SPARSE:
            assert(false);
            break;
          default:
            assert(false);
        }
        break;
      case kDOUBLE:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(double);
          case kENCODING_FIXED:
          case kENCODING_RL:
          case kENCODING_DIFF:
          case kENCODING_SPARSE:
            assert(false);
            break;
          default:
            assert(false);
        }
        break;
      case kTIMESTAMP:
      case kTIME:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
      case kDATE:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(int64_t);
          case kENCODING_FIXED:
            if (type == kTIMESTAMP && dimension > 0) {
              assert(false);  // disable compression for timestamp precisions
            }
            return comp_param / 8;
          case kENCODING_RL:
          case kENCODING_DIFF:
          case kENCODING_SPARSE:
            assert(false);
            break;
          case kENCODING_DATE_IN_DAYS:
            switch (comp_param) {
              case 0:
                return 4;  // Default date encoded in days is 32 bits
              case 16:
              case 32:
                return comp_param / 8;
              default:
                assert(false);
                break;
            }
          default:
            assert(false);
        }
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
        if (compression == kENCODING_DICT) {
          return sizeof(int32_t);  // @TODO(wei) must check DictDescriptor
        }
        break;
      case kARRAY:
        // TODO: return size for fixlen arrays?
        break;
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
      case kCOLUMN:
      case kCOLUMN_LIST:
        break;
      default:
        break;
    }
    return -1;
  }
};

SQLTypes decimal_to_int_type(const SQLTypeInfo&);

SQLTypes string_dict_to_int_type(const SQLTypeInfo&);

template <typename T>
SQLTypes cpp_type_to_sql_type() {
  if constexpr (std::is_same_v<int8_t, T>) {
    return kTINYINT;
  } else if constexpr (std::is_same_v<int16_t, T>) {
    return kSMALLINT;
  } else if constexpr (std::is_same_v<int32_t, T>) {
    return kINT;
  } else if constexpr (std::is_same_v<int64_t, T>) {
    return kBIGINT;
  } else if constexpr (std::is_same_v<float, T>) {
    return kFLOAT;
  } else if constexpr (std::is_same_v<double, T>) {
    return kDOUBLE;
  }
}

#ifndef __CUDACC__
#include <string_view>

Datum NullDatum(const SQLTypeInfo& ti);
bool IsNullDatum(const Datum d, const SQLTypeInfo& ti);
Datum StringToDatum(const std::string_view s, SQLTypeInfo& ti);
std::string DatumToString(const Datum d, const SQLTypeInfo& ti);
int64_t extract_int_type_from_datum(const Datum datum, const SQLTypeInfo& ti);
double extract_fp_type_from_datum(const Datum datum, const SQLTypeInfo& ti);
bool DatumEqual(const Datum, const Datum, const SQLTypeInfo& ti);
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info);
#endif

#ifdef HAVE_TOSTRING
inline std::ostream& operator<<(std::ostream& os, const SQLTypeInfo& type_info) {
  os << toString(type_info);
  return os;
}
#endif

#include "../QueryEngine/DateAdd.h"
#include "../QueryEngine/DateTruncate.h"
#include "../QueryEngine/ExtractFromTime.h"

inline SQLTypes get_int_type_by_size(size_t const nbytes) {
  switch (nbytes) {
    case 1:
      return kTINYINT;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
#if !(defined(__CUDACC__) || defined(NO_BOOST))
      UNREACHABLE() << "Invalid number of bytes=" << nbytes;
#endif
      return {};
  }
}

inline SQLTypeInfo get_logical_type_info(const SQLTypeInfo& type_info) {
  EncodingType encoding = type_info.get_compression();
  if (encoding == kENCODING_DATE_IN_DAYS ||
      (encoding == kENCODING_FIXED && type_info.get_type() != kARRAY)) {
    encoding = kENCODING_NONE;
  }
  auto type_info_copy = type_info;
  type_info_copy.set_compression(encoding);
  type_info_copy.setStorageSize();
  return type_info_copy;
}

inline SQLTypeInfo get_nullable_type_info(const SQLTypeInfo& type_info) {
  SQLTypeInfo nullable_type_info = type_info;
  nullable_type_info.set_notnull(false);
  return nullable_type_info;
}

inline SQLTypeInfo get_nullable_logical_type_info(const SQLTypeInfo& type_info) {
  SQLTypeInfo nullable_type_info = get_logical_type_info(type_info);
  return get_nullable_type_info(nullable_type_info);
}

using StringOffsetT = int32_t;
using ArrayOffsetT = int32_t;

int8_t* append_datum(int8_t* buf, const Datum& d, const SQLTypeInfo& ti);

// clang-format off
/*

A note on representing collection types using SQLTypeInfo
=========================================================

In general, a collection type is a type of collection of items. A
collection can be an array, a column, or a column list. A column list
is as collection of columns that have the same item type.  An item can
be of scalar type (bool, integers, floats, text encoding dict's, etc)
or of collection type (array of scalars, column of scalars, column of
array of scalars).

SQLTypeInfo provides a structure to represent both item and collection
types using the following list of attributes:
  SQLTypes type
  SQLTypes subtype
  int dimension
  int scale
  bool notnull
  EncodingType compression
  int comp_param
  int size

To represent a particular type, not all attributes are used. However,
there may exists multiple ways to represent the same type using
various combinations of these attributes and this note can be used as
a guideline to how to represent a newly introduced collection type
using the SQLTypeInfo structure.

Scalar types
------------

- Scalar types are booleans, integers, and floats that are defined
  by type and size attributes,

    SQLTypeInfo(type=kSCALAR)

  where SCALAR is in {BOOL, BIGINT, INT, SMALLINT, TINYINT, DOUBLE,
  FLOAT} while the corresponding size is specified in
  get_storage_size().  For example, SQLTypeInfo(type=kFLOAT)
  represents FLOAT and its size is implemented as 4 in the
  get_storage_size() method,

- Text encoding dict (as defined as index and string dictionary) is
  represented as a 32-bit integer value and its type is specified as

    SQLTypeInfo(type=kTEXT, compression=kENCODING_DICT, comp_param=<dict id>)

  and size is defined as 4 by get_storage_size().

Collection types
----------------

- The type of a varlen array of scalar items is specified as

    SQLTypeInfo(type=kARRAY, subtype=kSCALAR)

  and size is defined as -1 by get_storage_size() which can be interpreted as N/A.

- The type of a varlen array of text encoding dict is specified as

    SQLTypeInfo(type=kARRAY, subtype=kTEXT, compression=kENCODING_DICT, comp_param=<dict id>)

  Notice that the compression and comp_param attributes apply to
  subtype rather than to type. This quirk exemplifies the fact that
  SQLTypeInfo provides limited ability to support composite types.

- Similarly, the types of a column of scalar and text encoded dict
  items are specified as

    SQLTypeInfo(type=kCOLUMN, subtype=kSCALAR)

  and

    SQLTypeInfo(type=kCOLUMN, subtype=kTEXT, compression=kENCODING_DICT, comp_param=<dict id>)

  respectively.

- The type of column list with scalar items is specified as

    SQLTypeInfo(type=kCOLUMN_LIST, subtype=kSCALAR, dimension=<nof columns>)

  WARNING: Column list with items that type use compression (such as
  TIMESTAMP), cannot be supported! See QE-427.

- The type of column list with text encoded dict items is specified as

    SQLTypeInfo(type=kCOLUMN_LIST, subtype=kTEXT, compression=kENCODING_DICT, dimension=<nof columns>)

- The type of a column of arrays of scalar items is specified as

    SQLTypeInfo(type=kCOLUMN, subtype=kSCALAR, compression=kENCODING_ARRAY)

  Notice that the "a collection of collections of items" is specified
  by introducing a new compression scheme that descibes the
  "collections" part while the subtype attribute specifies the type of
  items.

- The type of a column of arrays of text encoding dict items is specified as

    SQLTypeInfo(type=kCOLUMN, subtype=kTEXT, compression=kENCODING_ARRAY_DICT, comp_param=<dict id>)

  where the compression attribute kENCODING_ARRAY_DICT carries two
  pieces of information: (i) the items type is dict encoded string and
  (ii) the type represents a "column of arrays".


- The type of a column list of arrays of scalar items is specified as

    SQLTypeInfo(type=kCOLUMN_LIST, subtype=kSCALAR, compression=kENCODING_ARRAY, dimension=<nof columns>)

- The type of a column list of arrays of text encoding dict items is specified as

    SQLTypeInfo(type=kCOLUMN_LIST, subtype=kTEXT, compression=kENCODING_ARRAY_DICT, comp_param=<dict id>, dimension=<nof columns>)

  that is the most complicated currently supported type of "a
  collection(=list) of collections(=columns) of collections(=arrays)
  of items(=text)" with a specified compression scheme and comp_param
  attributes.

- The type of a column of geometru points is specified as

    SQLTypeInfo(type=kCOLUMN, subtype=kPOINT, dimension=<input srid>, scale=<output srid>, compression=<kENCODING_NONE|kENCODING_GEOINT>)


*/
// clang-format on

inline auto generate_column_type(const SQLTypeInfo& elem_ti) {
  SQLTypes elem_type = elem_ti.get_type();
  if (elem_type == kCOLUMN) {
    if (elem_ti.get_subtype() == kVARCHAR) {
      auto new_elem_ti = elem_ti;
      new_elem_ti.set_subtype(kTEXT);
      return new_elem_ti;
    }
    return elem_ti;
  }
  auto c = elem_ti.get_compression();
  auto d = elem_ti.get_dimension();
  auto s = elem_ti.get_scale();
  auto p = elem_ti.get_comp_param();
  switch (elem_type) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kFLOAT:
    case kDOUBLE:
      if (c == kENCODING_NONE && p == 0) {
        break;  // here and below `break` means supported element type
                // for extension functions
      }
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOINT:
    case kMULTILINESTRING:
    case kMULTIPOLYGON:
      if (c == kENCODING_NONE && p == 0) {
        break;
      }
      if (c == kENCODING_GEOINT && p == 32) {
        break;
      }
    case kTEXT:
    case kVARCHAR:
      elem_type = kTEXT;
      if (c == kENCODING_DICT) {
        break;
      }
    case kTIMESTAMP:
      if (c == kENCODING_NONE && p == 0 && (d == 9 || d == 6 || d == 0)) {
        break;
      }
    case kARRAY:
      elem_type = elem_ti.get_subtype();
      if (IS_NUMBER(elem_type) || elem_type == kBOOLEAN || elem_type == kTEXT) {
        if (c == kENCODING_NONE && p == 0) {
          c = kENCODING_ARRAY;
          break;
        } else if (c == kENCODING_DICT) {
          c = kENCODING_ARRAY_DICT;
          break;
        }
      }
    default:
      elem_type = kNULLT;  // indicates unsupported element type that
                           // the caller needs to handle accordingly
  }
  auto ti = SQLTypeInfo(kCOLUMN, c, p, elem_type);
  ti.set_dimension(d);
  if (c == kENCODING_DICT) {
    ti.setStringDictKey(elem_ti.getStringDictKey());
  }
  ti.set_scale(s);
  return ti;
}

inline auto generate_column_list_type(const SQLTypeInfo& elem_ti) {
  auto type_info = generate_column_type(elem_ti);
  if (type_info.get_subtype() != kNULLT) {
    type_info.set_type(kCOLUMN_LIST);
  }
  if (type_info.get_subtype() == kTIMESTAMP) {
    // ColumnList<Timestamp> is not supported, see QE-472
    type_info.set_subtype(kNULLT);
  }
  return type_info;
}

// SQLTypeInfo-friendly interface to FlatBuffer:

// ChunkIter_get_nth variant for array buffers using FlatBuffer storage schema:
DEVICE inline void VarlenArray_get_nth(int8_t* buf,
                                       int n,
                                       ArrayDatum* result,
                                       bool* is_end) {
  FlatBufferManager m{buf};
  FlatBufferManager::Status status{};
  if (m.isNestedArray()) {
    FlatBufferManager::NestedArrayItem<1> item;
    status = m.getItem(n, item);
    if (status == FlatBufferManager::Status::Success) {
      result->length = item.nof_values * m.getValueSize();
      result->pointer = item.values;
      result->is_null = item.is_null;
      *is_end = false;
    } else {
      result->length = 0;
      result->pointer = NULL;
      result->is_null = true;
      *is_end = true;
#ifndef __CUDACC__
      // out of range indexing is equivalent to returning NULL (SQL AT)
      CHECK_EQ(status, FlatBufferManager::Status::ItemUnspecifiedError);
#endif
    }
  } else {
    // to be deprecated
    auto status = m.getItemOld(n, result->length, result->pointer, result->is_null);
    if (status == FlatBufferManager::Status::IndexError) {
      *is_end = true;
      result->length = 0;
      result->pointer = NULL;
      result->is_null = true;
    } else {
      *is_end = false;
#ifndef __CUDACC__
      CHECK_EQ(status, FlatBufferManager::Status::Success);
#endif
    }
  }
}

DEVICE inline void VarlenArray_get_nth(int8_t* buf,
                                       int n,
                                       bool uncompress,
                                       VarlenDatum* result,
                                       bool* is_end) {
  FlatBufferManager m{buf};
  FlatBufferManager::Status status{};
#ifndef __CUDACC__
  CHECK(m.isNestedArray());
#endif
  FlatBufferManager::NestedArrayItem<1> item;
  status = m.getItem(n, item);
  if (status == FlatBufferManager::Status::Success) {
    result->length = item.nof_values * m.getValueSize();
    result->pointer = item.values;
    result->is_null = item.is_null;
    *is_end = false;
#ifndef __CUDACC__
    CHECK(!uncompress);  // NOT IMPLEMENTED
#endif
  } else {
    result->length = 0;
    result->pointer = NULL;
    result->is_null = true;
    *is_end = true;
#ifndef __CUDACC__
    // out of range indexing is equivalent to returning NULL (SQL AT)
    CHECK_EQ(status, FlatBufferManager::Status::IndexError);
#endif
  }
}

inline void getFlatBufferNDimsAndSizes(const int64_t items_count,
                                       const int64_t max_nof_values,
                                       const SQLTypeInfo& ti,
                                       size_t& ndims,
                                       int64_t& max_nof_sizes) {
  ndims = 0;
  max_nof_sizes = 0;
  switch (ti.get_type()) {
    case kPOINT:
      ndims = 0;
      break;
    case kARRAY:
      if (ti.get_subtype() == kTEXT && ti.get_compression() == kENCODING_NONE) {
        ndims = 2;
        max_nof_sizes = items_count + 2 * max_nof_values / 3;
      } else {
        ndims = 1;
        max_nof_sizes = items_count + max_nof_values / 3;
      }
      break;
    case kLINESTRING:
    case kMULTIPOINT:
      ndims = 1;
      max_nof_sizes = items_count + max_nof_values / 3;
      break;
    case kPOLYGON:
    case kMULTILINESTRING:
      ndims = 2;
      max_nof_sizes = items_count + 2 * max_nof_values / 3;
      break;
    case kMULTIPOLYGON:
      ndims = 3;
      max_nof_sizes = items_count + max_nof_values;
      break;
    case kTEXT:
      switch (ti.get_compression()) {
        case kENCODING_NONE:
          ndims = 1;
          max_nof_sizes = items_count + max_nof_values / 3;
          break;
        case kENCODING_DICT:
          ndims = 0;
          break;
        default:
          UNREACHABLE();
      }
      break;
    default:
      UNREACHABLE();
  }
}

inline int64_t getFlatBufferSize(int64_t items_count,
                                 int64_t max_nof_values,
                                 const SQLTypeInfo& ti) {
  if (ti.get_type() == kPOINT) {
    // to be deprecated
    FlatBufferManager::GeoPoint metadata{items_count,
                                         ti.get_input_srid(),
                                         ti.get_output_srid(),
                                         ti.get_compression() == kENCODING_GEOINT};
    return FlatBufferManager::compute_flatbuffer_size(
        GeoPointFormatId, reinterpret_cast<const int8_t*>(&metadata));
  } else {
    size_t ndims = 0;
    int64_t max_nof_sizes = 0;
    getFlatBufferNDimsAndSizes(items_count, max_nof_values, ti, ndims, max_nof_sizes);
    return FlatBufferManager::computeBufferSizeNestedArray(
        /* ndims= */ ndims,
        /* total_items_count= */ items_count,
        /* total sizes count= */ max_nof_sizes,
        /* total values count= */ max_nof_values,
        ti.toValueType(),
        /* user data size= */ sizeof(SQLTypeInfoLite));
  }
}

typedef union {
  struct {
    int8_t i8;
  };
  struct {
    int16_t i16;
  };
  struct {
    int32_t i32;
  };
  struct {
    int64_t i64;
  };
  struct {
    float f32;
  };
  struct {
    double f64;
  };
  struct {
    uint32_t geoint[2];
  };
  struct {
    double geodouble[2];
  };
} null_value_t;

inline null_value_t get_null_value(const SQLTypeInfo& ti) {
  null_value_t null_value{};
  switch (ti.get_type()) {
    case kBOOLEAN:
      null_value.i8 = NULL_BOOLEAN;
      break;
    case kTINYINT:
      null_value.i8 = NULL_TINYINT;
      break;
    case kSMALLINT:
      null_value.i16 = NULL_SMALLINT;
      break;
    case kINT:
      null_value.i32 = NULL_INT;
      break;
    case kBIGINT:
      null_value.i64 = NULL_BIGINT;
      break;
    case kFLOAT:
      null_value.f32 = NULL_FLOAT;
      break;
    case kDOUBLE:
      null_value.f64 = NULL_DOUBLE;
      break;
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      if (ti.get_compression() == kENCODING_GEOINT) {
        null_value.geoint[0] = null_value.geoint[1] = NULL_ARRAY_COMPRESSED_32;
      } else if (ti.get_compression() == kENCODING_NONE) {
        null_value.geodouble[0] = null_value.geodouble[1] = NULL_ARRAY_DOUBLE;
      } else {
        UNREACHABLE();
      }
      break;
    case kARRAY:
      switch (ti.get_subtype()) {
        case kBOOLEAN:
          null_value.i8 = NULL_ARRAY_BOOLEAN;
          break;
        case kTINYINT:
          null_value.i8 = NULL_ARRAY_TINYINT;
          break;
        case kSMALLINT:
          null_value.i16 = NULL_ARRAY_SMALLINT;
          break;
        case kINT:
          null_value.i32 = NULL_ARRAY_INT;
          break;
        case kTIMESTAMP:
        case kTIME:
        case kDATE:
        case kINTERVAL_DAY_TIME:
        case kINTERVAL_YEAR_MONTH:
        case kDECIMAL:
        case kNUMERIC:
        case kBIGINT:
          null_value.i64 = NULL_ARRAY_BIGINT;
          break;
        case kFLOAT:
          null_value.f32 = NULL_ARRAY_FLOAT;
          break;
        case kDOUBLE:
          null_value.f64 = NULL_ARRAY_FLOAT;
          break;
        case kTEXT:
          if (ti.get_compression() == kENCODING_DICT) {
            CHECK_EQ(ti.get_logical_size(), 4);
            null_value.i32 = NULL_ARRAY_COMPRESSED_32;
          } else if (ti.get_compression() == kENCODING_NONE) {
            null_value.i8 = NULL_ARRAY_TINYINT;
          } else {
            UNREACHABLE();
          }
          break;
        default:
          UNREACHABLE();
          break;
      }
      break;
    case kTEXT:
      if (ti.get_compression() == kENCODING_DICT) {
        CHECK_EQ(ti.get_logical_size(), 4);
        null_value.i32 = NULL_INT;
      } else if (ti.get_compression() == kENCODING_NONE) {
        null_value.i32 = NULL_TINYINT;
      } else {
        UNREACHABLE();
      }
      break;
    default:
      UNREACHABLE();
      break;
  }
  return null_value;
}

inline void initializeFlatBuffer(FlatBufferManager& m,
                                 int64_t items_count,
                                 int64_t max_nof_values,
                                 const SQLTypeInfo& ti) {
  if (ti.get_type() == kPOINT) {
    // to be deprecated
    FlatBufferManager::GeoPoint metadata{items_count,
                                         ti.get_input_srid(),
                                         ti.get_output_srid(),
                                         ti.get_compression() == kENCODING_GEOINT};
    m.initialize(GeoPointFormatId, reinterpret_cast<const int8_t*>(&metadata));
  } else {
    size_t ndims = 0;
    int64_t max_nof_sizes = 0;
    getFlatBufferNDimsAndSizes(items_count, max_nof_values, ti, ndims, max_nof_sizes);
    SQLTypeInfoLite ti_lite = ti.toLite();
    null_value_t null_value = get_null_value(ti);
    int8_t* null_value_ptr = &null_value.i8;
    auto status = m.initializeNestedArray(
        /* ndims= */ ndims,
        /* total_items_count= */ items_count,
        /* total_sizes_count= */ max_nof_sizes,
        /* total_values_count= */ max_nof_values,
        ti.toValueType(),
        /* null value buffer=*/null_value_ptr,  // null value buffer size
                                                // is defined by value type
        /* user data buffer=*/reinterpret_cast<const int8_t*>(&ti_lite),
        /* user data buffer size=*/sizeof(SQLTypeInfoLite));
    CHECK_EQ(status, FlatBufferManager::Success);
  }
}

struct SqlLiteralArg {
  size_t index;
  SQLTypeInfo ti;
  Datum datum;
};

inline bool geo_promoted_type_match(const SQLTypes a, const SQLTypes b) {
  return (a == b) || (a == kPOINT && b == kMULTIPOINT) ||
         (a == kLINESTRING && b == kMULTILINESTRING) ||
         (a == kPOLYGON && b == kMULTIPOLYGON);
}
