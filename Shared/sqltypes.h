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

/**
 * @file		sqltypes.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Constants for Builtin SQL Types supported by MapD
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef SQLTYPES_H
#define SQLTYPES_H

#include "funcannotations.h"

#include <stdint.h>
#include <ctime>
#include <cfloat>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cassert>

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
  kSQLTYPE_LAST = 18
};

struct VarlenDatum {
  int length;
  int8_t* pointer;
  bool is_null;

  DEVICE VarlenDatum() : length(0), pointer(NULL), is_null(true) {}
  VarlenDatum(int l, int8_t* p, bool n) : length(l), pointer(p), is_null(n) {}
};

// ArrayDatum is idential to VarlenDatum except that it takes ownership of
// the memory holding array data.
struct ArrayDatum {
  size_t length;
  int8_t* pointer;
  bool is_null;
#ifndef __CUDACC__
  std::shared_ptr<int8_t> data_ptr;
#endif

  DEVICE ArrayDatum() : length(0), pointer(NULL), is_null(true) {}
#ifndef __CUDACC__
  ArrayDatum(int l, int8_t* p, bool n) : length(l), pointer(p), is_null(n), data_ptr(p, [](int8_t* p) { free(p); }) {}
#else
  ArrayDatum(int l, int8_t* p, bool n) : length(l), pointer(p), is_null(n) {}
#endif
};

typedef union {
  bool boolval;
  int8_t tinyintval;
  int16_t smallintval;
  int32_t intval;
  int64_t bigintval;
  std::time_t timeval;
  float floatval;
  double doubleval;
  VarlenDatum* arrayval;
#ifndef __CUDACC__
  std::string* stringval;  // string value
#endif
} Datum;

#ifndef __CUDACC__
union DataBlockPtr {
  int8_t* numbersPtr;
  std::vector<std::string>* stringsPtr;
  std::vector<ArrayDatum>* arraysPtr;
};
#endif

// must not change because these values persist in catalogs.
enum EncodingType {
  kENCODING_NONE = 0,    // no encoding
  kENCODING_FIXED = 1,   // Fixed-bit encoding
  kENCODING_RL = 2,      // Run Length encoding
  kENCODING_DIFF = 3,    // Differential encoding
  kENCODING_DICT = 4,    // Dictionary encoding
  kENCODING_SPARSE = 5,  // Null encoding for sparse columns
  kENCODING_LAST = 6
};

#define IS_INTEGER(T) (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kBIGINT))
#define IS_NUMBER(T)                                                                                 \
  (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kDOUBLE) || ((T) == kFLOAT) || ((T) == kBIGINT) || \
   ((T) == kNUMERIC) || ((T) == kDECIMAL))
#define IS_STRING(T) (((T) == kTEXT) || ((T) == kVARCHAR) || ((T) == kCHAR))
#define IS_TIME(T) (((T) == kTIME) || ((T) == kTIMESTAMP) || ((T) == kDATE))

#define NULL_BOOLEAN INT8_MIN
#define NULL_TINYINT INT8_MIN
#define NULL_SMALLINT INT16_MIN
#define NULL_INT INT32_MIN
#define NULL_BIGINT INT64_MIN
#define NULL_FLOAT FLT_MIN
#define NULL_DOUBLE DBL_MIN

#define TRANSIENT_DICT_ID 0
#define TRANSIENT_DICT(ID) (-(ID))
#define REGULAR_DICT(TRANSIENTID) (-(TRANSIENTID))

// @type SQLTypeInfo
// @brief a structure to capture all type information including
// length, precision, scale, etc.
class SQLTypeInfo {
 public:
  SQLTypeInfo(SQLTypes t, int d, int s, bool n, EncodingType c, int p, SQLTypes st)
      : type(t),
        subtype(st),
        dimension(d),
        scale(s),
        notnull(n),
        compression(c),
        comp_param(p),
        size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t, int d, int s, bool n)
      : type(t),
        subtype(kNULLT),
        dimension(d),
        scale(s),
        notnull(n),
        compression(kENCODING_NONE),
        comp_param(0),
        size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t, bool n)
      : type(t),
        subtype(kNULLT),
        dimension(0),
        scale(0),
        notnull(n),
        compression(kENCODING_NONE),
        comp_param(0),
        size(get_storage_size()) {}
  SQLTypeInfo(SQLTypes t, bool n, EncodingType c)
      : type(t),
        subtype(kNULLT),
        dimension(0),
        scale(0),
        notnull(n),
        compression(c),
        comp_param(0),
        size(get_storage_size()) {}
  SQLTypeInfo()
      : type(kNULLT),
        subtype(kNULLT),
        dimension(0),
        scale(0),
        notnull(false),
        compression(kENCODING_NONE),
        comp_param(0),
        size(0) {}

  HOST DEVICE inline SQLTypes get_type() const { return type; }
  HOST DEVICE inline SQLTypes get_subtype() const { return subtype; }
  HOST DEVICE inline int get_dimension() const { return dimension; }
  inline int get_precision() const { return dimension; }
  HOST DEVICE inline int get_scale() const { return scale; }
  HOST DEVICE inline bool get_notnull() const { return notnull; }
  HOST DEVICE inline EncodingType get_compression() const { return compression; }
  HOST DEVICE inline int get_comp_param() const { return comp_param; }
  HOST DEVICE inline int get_size() const { return size; }
  inline int get_logical_size() const {
    if (compression == kENCODING_FIXED) {
      SQLTypeInfo ti(type, dimension, scale, notnull, kENCODING_NONE, 0, subtype);
      return ti.get_size();
    }
    if (compression == kENCODING_DICT) {
      return 4;
    }
    return get_size();
  }
  inline void set_type(SQLTypes t) { type = t; }
  inline void set_subtype(SQLTypes st) { subtype = st; }
  inline void set_dimension(int d) { dimension = d; }
  inline void set_precision(int d) { dimension = d; }
  inline void set_scale(int s) { scale = s; }
  inline void set_notnull(bool n) { notnull = n; }
  inline void set_size(int s) { size = s; }
  inline void set_fixed_size() { size = get_storage_size(); }
  inline void set_compression(EncodingType c) { compression = c; }
  inline void set_comp_param(int p) { comp_param = p; }
#ifndef __CUDACC__
  inline std::string get_type_name() const {
    std::string ps = (type == kDECIMAL || type == kNUMERIC || subtype == kDECIMAL || subtype == kNUMERIC)
                         ? "(" + std::to_string(dimension) + "," + std::to_string(scale) + ")"
                         : "";
    return (type == kARRAY) ? type_name[(int)subtype] + ps + "[]" : type_name[(int)type] + ps;
  }
  inline std::string get_compression_name() const { return comp_name[(int)compression]; }
#endif
  inline bool is_string() const { return IS_STRING(type); }
  inline bool is_string_array() const { return (type == kARRAY) && IS_STRING(subtype); }
  inline bool is_integer() const { return IS_INTEGER(type); }
  inline bool is_decimal() const { return type == kDECIMAL || type == kNUMERIC; }
  inline bool is_fp() const { return type == kFLOAT || type == kDOUBLE; }
  inline bool is_number() const { return IS_NUMBER(type); }
  inline bool is_time() const { return IS_TIME(type); }
  inline bool is_boolean() const { return type == kBOOLEAN; }
  inline bool is_array() const { return type == kARRAY; }
  inline bool is_timeinterval() const { return type == kINTERVAL_DAY_TIME || type == kINTERVAL_YEAR_MONTH; }

  inline bool is_varlen() const { return (IS_STRING(type) && compression != kENCODING_DICT) || type == kARRAY; }

  HOST DEVICE inline bool operator!=(const SQLTypeInfo& rhs) const {
    return type != rhs.get_type() || subtype != rhs.get_subtype() || dimension != rhs.get_dimension() ||
           scale != rhs.get_scale() || compression != rhs.get_compression() ||
           (compression != kENCODING_NONE && comp_param != rhs.get_comp_param() &&
            comp_param != TRANSIENT_DICT(rhs.get_comp_param())) ||
           notnull != rhs.get_notnull();
  }
  HOST DEVICE inline bool operator==(const SQLTypeInfo& rhs) const {
    return type == rhs.get_type() && subtype == rhs.get_subtype() && dimension == rhs.get_dimension() &&
           scale == rhs.get_scale() && compression == rhs.get_compression() &&
           (compression == kENCODING_NONE || comp_param == rhs.get_comp_param() ||
            comp_param == TRANSIENT_DICT(rhs.get_comp_param())) &&
           notnull == rhs.get_notnull();
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
    if (type == new_type_info.get_type())
      return true;
    // can always cast from or to string
    else if (is_string() || new_type_info.is_string())
      return true;
    // can cast between numbers
    else if (is_number() && new_type_info.is_number())
      return true;
    // can cast from timestamp or date to number (epoch)
    else if ((type == kTIMESTAMP || type == kDATE) && new_type_info.is_number())
      return true;
    // can cast from date to timestamp
    else if (type == kDATE && new_type_info.get_type() == kTIMESTAMP)
      return true;
    else if (type == kTIMESTAMP && new_type_info.get_type() == kDATE)
      return true;
    else if (type == kBOOLEAN && new_type_info.is_number())
      return true;
    else if (type == kARRAY && new_type_info.get_type() == kARRAY)
      return get_elem_type().is_castable(new_type_info.get_elem_type());
    else
      return false;
  }
  HOST DEVICE inline bool is_null(const Datum& d) const {
    // assuming Datum is always uncompressed
    switch (type) {
      case kBOOLEAN:
        return (int8_t)d.boolval == NULL_BOOLEAN;
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
// @TODO(alex): remove the ifdef
#ifdef __ARM_ARCH_7A__
#ifndef __CUDACC__
        static_assert(sizeof(time_t) == 4, "Unsupported time_t size");
#endif
        return d.timeval == NULL_INT;
#else
#ifndef __CUDACC__
        static_assert(sizeof(time_t) == 8, "Unsupported time_t size");
#endif
        return d.timeval == NULL_BIGINT;
#endif
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
  inline SQLTypeInfo get_elem_type() const {
    return SQLTypeInfo(subtype, dimension, scale, notnull, compression, comp_param, kNULLT);
  }
  inline SQLTypeInfo get_array_type() const {
    return SQLTypeInfo(kARRAY, dimension, scale, notnull, compression, comp_param, type);
  }

 private:
  SQLTypes type;             // type id
  SQLTypes subtype;          // element type of arrays
  int dimension;             // VARCHAR/CHAR length or NUMERIC/DECIMAL precision
  int scale;                 // NUMERIC/DECIMAL scale
  bool notnull;              // nullable?  a hint, not used for type checking
  EncodingType compression;  // compression scheme
  int comp_param;            // compression parameter when applicable for certain schemes
  int size;                  // size of the type in bytes.  -1 for variable size
#ifndef __CUDACC__
  static std::string type_name[kSQLTYPE_LAST];
  static std::string comp_name[kENCODING_LAST];
#endif
  inline int get_storage_size() const {
    switch (type) {
      case kBOOLEAN:
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
      case kTIME:
      case kTIMESTAMP:
        if (dimension > 0)
          assert(false);  // not supported yet
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
      case kDATE:
        switch (compression) {
          case kENCODING_NONE:
            return sizeof(time_t);
          case kENCODING_FIXED:
            return comp_param / 8;
          case kENCODING_RL:
          case kENCODING_DIFF:
          case kENCODING_SPARSE:
            assert(false);
            break;
          default:
            assert(false);
        }
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
        if (compression == kENCODING_DICT)
          return sizeof(int32_t);  // @TODO(wei) must check DictDescriptor
        break;
      default:
        break;
    }
    return -1;
  }
};

SQLTypes decimal_to_int_type(const SQLTypeInfo&);

#ifndef __CUDACC__
Datum StringToDatum(const std::string& s, SQLTypeInfo& ti);
std::string DatumToString(Datum d, const SQLTypeInfo& ti);
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info);
#endif

#include "../QueryEngine/ExtractFromTime.h"
#include "../QueryEngine/DateTruncate.h"
#include "../QueryEngine/DateAdd.h"

inline SQLTypeInfo get_logical_type_info(const SQLTypeInfo& type_info) {
  EncodingType encoding = type_info.get_compression();
  if (encoding == kENCODING_FIXED) {
    encoding = kENCODING_NONE;
  }
  return SQLTypeInfo(type_info.get_type(),
                     type_info.get_dimension(),
                     type_info.get_scale(),
                     false,
                     encoding,
                     0,
                     type_info.get_subtype());
}

template <class T>
inline int64_t inline_int_null_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
}

template <class T>
inline int64_t max_valid_int_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::max() : std::numeric_limits<T>::max() - 1;
}

template <typename T>
T inline_fp_null_value();

template <>
inline float inline_fp_null_value<float>() {
  return NULL_FLOAT;
}

template <>
inline double inline_fp_null_value<double>() {
  return NULL_DOUBLE;
}

typedef int32_t StringOffsetT;

#endif  // SQLTYPES_H
