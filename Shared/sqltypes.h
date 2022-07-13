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

#include "../Logger/Logger.h"
#include "funcannotations.h"
#include "toString.h"

#include <cassert>
#include <ctime>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

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
  kSQLTYPE_LAST = 31
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

struct VarlenDatum {
  size_t length;
  int8_t* pointer;
  bool is_null;

  DEVICE VarlenDatum() : length(0), pointer(nullptr), is_null(true) {}
  DEVICE virtual ~VarlenDatum() {}

  VarlenDatum(const size_t l, int8_t* p, const bool n)
      : length(l), pointer(p), is_null(n) {}
};

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
                std::is_void<std::result_of_t<CUSTOM_DELETER(int8_t*)> >::value> >
  HostArrayDatum(size_t const l, int8_t* p, CUSTOM_DELETER custom_deleter)
      : VarlenDatum(l, p, 0 == l), data_ptr(p, custom_deleter) {}

  template <typename CUSTOM_DELETER,
            typename = std::enable_if_t<
                std::is_void<std::result_of_t<CUSTOM_DELETER(int8_t*)> >::value> >
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

union Datum {
  int8_t boolval;
  int8_t tinyintval;
  int16_t smallintval;
  int32_t intval;
  int64_t bigintval;
  float floatval;
  double doubleval;
  VarlenDatum* arrayval;
#ifndef __CUDACC__
  std::string* stringval;  // string value
#endif
};

#ifndef __CUDACC__
union DataBlockPtr {
  int8_t* numbersPtr;
  std::vector<std::string>* stringsPtr;
  std::vector<ArrayDatum>* arraysPtr;
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
  kENCODING_LAST = 9
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
   ((T) == kPOLYGON) || ((T) == kMULTIPOLYGON))
#define IS_INTERVAL(T) ((T) == kINTERVAL_DAY_TIME || (T) == kINTERVAL_YEAR_MONTH)
#define IS_DECIMAL(T) ((T) == kNUMERIC || (T) == kDECIMAL)
#define IS_GEO_POLY(T) (((T) == kPOLYGON) || ((T) == kMULTIPOLYGON))
#define IS_GEO_LINE(T) (((T) == kLINESTRING) || ((T) == kMULTILINESTRING))
#define IS_GEO_MULTI(T) (((T) == kMULTIPOLYGON) || ((T) == kMULTILINESTRING))

#include "InlineNullValues.h"

#define TRANSIENT_DICT_ID 0
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
  HOST DEVICE inline int get_comp_param() const { return comp_param; }
  HOST DEVICE inline int get_size() const { return size; }

  inline int is_logical_geo_type() const {
    if (type == kPOINT || type == kLINESTRING || type == kMULTILINESTRING ||
        type == kPOLYGON || type == kMULTIPOLYGON) {
      return true;
    }
    return false;
  }

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
      case kLINESTRING:
        return 2;  // coords, bounds
      case kMULTILINESTRING:
        return 3;  // coords, linestring_sizes, bounds
      case kPOLYGON:
        return 4;  // coords, ring_sizes, bounds, render_group
      case kMULTIPOLYGON:
        return 5;  // coords, ring_sizes, poly_rings, bounds, render_group
      default:
        break;
    }
    return 0;
  }
  inline int get_physical_coord_cols() const {
    // @TODO dmitri/simon rename this function?
    // It needs to return the number of extra columns
    // which need to go through the executor, as opposed
    // to those which are only needed by CPU for poly
    // cache building or what-not. For now, we just omit
    // the Render Group column. If we add Bounding Box
    // or something this may require rethinking. Perhaps
    // these two functions need to return an array of
    // offsets rather than just a number to loop over,
    // so that executor and non-executor columns can
    // be mixed.
    // NOTE(adb): In binding to extension functions, we need to know some pretty specific
    // type info about each of the physical coords cols for each geo type. I added checks
    // there to ensure the physical coords col for the geo type match what we expect. If
    // these values are ever changed, corresponding values in
    // ExtensionFunctionsBinding.cpp::compute_narrowing_conv_scores and
    // ExtensionFunctionsBinding.cpp::compute_widening_conv_scores will also need to be
    // changed.
    switch (type) {
      case kPOINT:
        return 1;
      case kLINESTRING:
        return 1;  // omit bounds
      case kMULTILINESTRING:
        return 2;  // omit bounds
      case kPOLYGON:
        return 2;  // omit bounds, render group
      case kMULTIPOLYGON:
        return 3;  // omit bounds, render group
      default:
        break;
    }
    return 0;
  }
  inline bool has_bounds() const {
    switch (type) {
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
  inline bool has_render_group() const {
    switch (type) {
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
  inline std::string get_compression_name() const { return comp_name[(int)compression]; }
  std::string toString() const { return to_string(); }  // for PRINT macro
  inline std::string to_string() const {
    std::ostringstream oss;
    oss << "(type=" << type_name[static_cast<int>(type)]
        << ", dimension=" << get_dimension() << ", scale=" << get_scale()
        << ", null=" << (get_notnull() ? "not nullable" : "nullable")
        << ", compression_name=" << get_compression_name()
        << ", comp_param=" << get_comp_param()
        << ", subtype=" << type_name[static_cast<int>(subtype)] << ", size=" << get_size()
        << ", element_size=" << get_elem_type().get_size() << ")";
    return oss.str();
  }

  inline std::string get_buffer_name() const {
    if (is_array()) {
      return "Array";
    }
    if (is_bytes()) {
      return "Bytes";
    }

    if (is_column()) {
      return "Column";
    }

    assert(false);
    return "";
  }
#endif
  template <typename... Types>
  bool is_any(Types... types) const {
    return (... || (types == type));
  }
  inline bool is_string() const { return IS_STRING(type); }
  inline bool is_string_array() const { return (type == kARRAY) && IS_STRING(subtype); }
  inline bool is_integer() const { return IS_INTEGER(type); }
  inline bool is_decimal() const { return type == kDECIMAL || type == kNUMERIC; }
  inline bool is_fp() const { return type == kFLOAT || type == kDOUBLE; }
  inline bool is_number() const { return IS_NUMBER(type); }
  inline bool is_time() const { return is_datetime(type); }
  inline bool is_boolean() const { return type == kBOOLEAN; }
  inline bool is_array() const { return type == kARRAY; }  // Array
  inline bool is_varlen_array() const { return type == kARRAY && size <= 0; }
  inline bool is_fixlen_array() const { return type == kARRAY && size > 0; }
  inline bool is_timeinterval() const { return IS_INTERVAL(type); }
  inline bool is_geometry() const { return IS_GEO(type); }
  inline bool is_column() const { return type == kCOLUMN; }            // Column
  inline bool is_column_list() const { return type == kCOLUMN_LIST; }  // ColumnList
  inline bool is_column_array() const {
    return type == kCOLUMN && get_compression() == kENCODING_ARRAY;
  }  // ColumnArray
  inline bool is_column_list_array() const {
    return type == kCOLUMN_LIST && get_compression() == kENCODING_ARRAY;
  }  // ColumnList of ColumnArray
  inline bool is_bytes() const {
    return type == kTEXT && get_compression() == kENCODING_NONE;
  }  // rbc Bytes
  inline bool is_buffer() const {
    return is_array() || is_column() || is_column_list() || is_bytes();
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

  inline bool is_dict_intersection() const { return dict_intersection; }

  inline bool has_same_itemtype(const SQLTypeInfo& other) const {
    if ((is_column() || is_column_list()) &&
        (other.is_column() || other.is_column_list())) {
      return subtype == other.get_subtype() &&
             (compression != kENCODING_ARRAY || compression == other.get_compression());
    }
    return subtype == other.get_subtype();
  }

  HOST DEVICE inline bool operator!=(const SQLTypeInfo& rhs) const {
    return type != rhs.get_type() || subtype != rhs.get_subtype() ||
           dimension != rhs.get_dimension() || scale != rhs.get_scale() ||
           compression != rhs.get_compression() ||
           (compression != kENCODING_NONE && comp_param != rhs.get_comp_param() &&
            comp_param != TRANSIENT_DICT(rhs.get_comp_param())) ||
           notnull != rhs.get_notnull();
  }
  HOST DEVICE inline bool operator==(const SQLTypeInfo& rhs) const {
    return type == rhs.get_type() && subtype == rhs.get_subtype() &&
           dimension == rhs.get_dimension() && scale == rhs.get_scale() &&
           compression == rhs.get_compression() &&
           (compression == kENCODING_NONE || comp_param == rhs.get_comp_param() ||
            comp_param == TRANSIENT_DICT(rhs.get_comp_param())) &&
           notnull == rhs.get_notnull();
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
      return false;
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
    if ((type == kCOLUMN || type == kCOLUMN_LIST) && compression == kENCODING_ARRAY) {
      return SQLTypeInfo(
          kARRAY, dimension, scale, notnull, kENCODING_NONE, comp_param, subtype);
    }
    return SQLTypeInfo(
        subtype, dimension, scale, notnull, compression, comp_param, kNULLT);
  }
  inline SQLTypeInfo get_array_type() const {
    return SQLTypeInfo(kARRAY, dimension, scale, notnull, compression, comp_param, type);
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

  inline bool is_date() const { return type == kDATE; }

  inline bool is_high_precision_timestamp() const {
    if (type == kTIMESTAMP) {
      const auto dimension = get_dimension();
      if (dimension > 0) {
        return true;
      }
    }
    return false;
  }

  inline bool is_timestamp() const { return type == kTIMESTAMP; }

 private:
  SQLTypes type;     // type id
  SQLTypes subtype;  // element type of arrays or columns
  int dimension;     // VARCHAR/CHAR length or NUMERIC/DECIMAL precision or COLUMN_LIST
                     // length
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

#ifndef __CUDACC__
#include <string_view>

Datum StringToDatum(std::string_view s, SQLTypeInfo& ti);
std::string DatumToString(Datum d, const SQLTypeInfo& ti);
int64_t extract_int_type_from_datum(const Datum datum, const SQLTypeInfo& ti);
double extract_fp_type_from_datum(const Datum datum, const SQLTypeInfo& ti);
bool DatumEqual(const Datum, const Datum, const SQLTypeInfo& ti);
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info);
#endif

#if !(defined(__CUDACC__) || defined(NO_BOOST))
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
  return SQLTypeInfo(type_info.get_type(),
                     type_info.get_dimension(),
                     type_info.get_scale(),
                     type_info.get_notnull(),
                     encoding,
                     type_info.get_comp_param(),
                     type_info.get_subtype());
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

inline auto generate_array_type(const SQLTypes subtype) {
  auto ti = SQLTypeInfo(kARRAY, false);
  ti.set_subtype(subtype);
  return ti;
}

inline auto generate_column_type(const SQLTypes subtype) {
  auto ti = SQLTypeInfo(kCOLUMN, false);
  ti.set_subtype(subtype);
  return ti;
}

inline auto generate_column_type(const SQLTypes subtype, EncodingType c, int p) {
  auto ti = SQLTypeInfo(kCOLUMN, false);
  ti.set_subtype(subtype);
  ti.set_compression(c);
  ti.set_comp_param(p);
  return ti;
}

inline auto generate_column_array_type(const SQLTypes subtype) {
  auto ti = SQLTypeInfo(kCOLUMN, false, kENCODING_ARRAY);
  ti.set_subtype(subtype);
  return ti;
}

inline auto generate_column_list_type(const SQLTypes subtype) {
  auto ti = SQLTypeInfo(kCOLUMN_LIST, false);
  ti.set_subtype(subtype);
  return ti;
}

inline auto generate_column_list_array_type(const SQLTypes subtype) {
  auto ti = SQLTypeInfo(kCOLUMN_LIST, false, kENCODING_ARRAY);
  ti.set_subtype(subtype);
  return ti;
}

struct SqlLiteralArg {
  size_t index;
  SQLTypeInfo ti;
  Datum datum;
};
