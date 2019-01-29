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

#include "ConfigResolve.h"

#include <stdint.h>
#include <cassert>
#include <cfloat>
#include <ctime>
#include <limits>
#include <memory>
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
  kSQLTYPE_LAST = 26
};

struct VarlenDatum {
  size_t length;
  int8_t* pointer;
  bool is_null;

  DEVICE VarlenDatum() : length(0), pointer(NULL), is_null(true) {}
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

  HostArrayDatum(size_t const l, int8_t* p, bool const n)
      : VarlenDatum(l, p, n), data_ptr(p, FreeDeleter()){};

  template <typename CUSTOM_DELETER,
            typename = std::enable_if_t<
                std::is_void<std::result_of_t<CUSTOM_DELETER(int8_t*)> >::value> >
  HostArrayDatum(size_t const l, int8_t* p, CUSTOM_DELETER custom_deleter)
      : VarlenDatum(l, p, 0 == l), data_ptr(p, custom_deleter) {}

  ManagedPtr data_ptr;
};

struct DeviceArrayDatum : public VarlenDatum {
  DEVICE DeviceArrayDatum() : VarlenDatum() {}
};

using ArrayDatum = std::conditional_t<isCudaCC(), DeviceArrayDatum, HostArrayDatum>;

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
  kENCODING_NONE = 0,          // no encoding
  kENCODING_FIXED = 1,         // Fixed-bit encoding
  kENCODING_RL = 2,            // Run Length encoding
  kENCODING_DIFF = 3,          // Differential encoding
  kENCODING_DICT = 4,          // Dictionary encoding
  kENCODING_SPARSE = 5,        // Null encoding for sparse columns
  kENCODING_GEOINT = 6,        // Encoding coordinates as intergers
  kENCODING_DATE_IN_DAYS = 7,  // Date encoding in days
  kENCODING_LAST = 8
};

#include "SQLTypeUtilities.h"

#define IS_INTEGER(T) \
  (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kBIGINT) || ((T) == kTINYINT))
#define IS_NUMBER(T)                                                             \
  (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kDOUBLE) || ((T) == kFLOAT) || \
   ((T) == kBIGINT) || ((T) == kNUMERIC) || ((T) == kDECIMAL) || ((T) == kTINYINT))
#define IS_STRING(T) (((T) == kTEXT) || ((T) == kVARCHAR) || ((T) == kCHAR))
#define IS_TIME(T) (((T) == kTIME) || ((T) == kTIMESTAMP) || ((T) == kDATE))
#define IS_GEO(T) \
  (((T) == kPOINT) || ((T) == kLINESTRING) || ((T) == kPOLYGON) || ((T) == kMULTIPOLYGON))
#define IS_INTERVAL(T) ((T) == kINTERVAL_DAY_TIME || (T) == kINTERVAL_YEAR_MONTH)
#define IS_DECIMAL(T) ((T) == kNUMERIC || (T) == kDECIMAL)
#define IS_GEO_POLY(T) (((T) == kPOLYGON) || ((T) == kMULTIPOLYGON))

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

template <typename CORE_TYPE>
class ExecutorTypePackaging {
 public:
  enum PackagingType { Chunk, StandardBuffer };

  ExecutorTypePackaging() : packaging_type_(Chunk) {}

  bool isStandardBufferPackaging() const { return packaging_type_ == StandardBuffer; }
  bool isChunkIteratorPackaging() const { return packaging_type_ == Chunk; }
  void setStandardBufferPackaging() { packaging_type_ = StandardBuffer; }
  void setChunkIteratorPackaging() { packaging_type_ = Chunk; }

 private:
  PackagingType packaging_type_;
};

template <typename CORE_TYPE>
class ArrayContextTypeSizer {
 public:
  inline int get_array_context_logical_size() const {
    CORE_TYPE const* derived(static_cast<CORE_TYPE const*>(this));
    if (is_member_of_typeset<kCHAR, kTEXT, kVARCHAR>(*derived)) {
      auto comp_type(derived->get_compression());
      if (comp_type == kENCODING_DICT || comp_type == kENCODING_FIXED ||
          comp_type == kENCODING_NONE) {
        return sizeof(int32_t);
      }
    }
    return derived->get_logical_size();
  }
};

template <typename CORE_TYPE>
class DateTimeFacilities {
 public:
  inline auto is_date_in_days() const {
    CORE_TYPE const* derived(static_cast<CORE_TYPE const*>(this));
    if (is_member_of_typeset<kDATE>(*derived)) {
      auto comp_type(derived->get_compression());
      if (comp_type == kENCODING_DATE_IN_DAYS) {
        return true;
      }
    }
    return false;
  }

  inline auto is_high_precision_timestamp() const {
    CORE_TYPE const* derived(static_cast<CORE_TYPE const*>(this));
    if (is_member_of_typeset<kTIMESTAMP>(*derived)) {
      auto dimension(derived->get_dimension());
      if (dimension > 0) {
        return true;
      }
    }
    return false;
  }

  inline auto is_timestamp() const {
    CORE_TYPE const* derived(static_cast<CORE_TYPE const*>(this));
    if (is_member_of_typeset<kTIMESTAMP>(*derived)) {
      return true;
    }
    return false;
  }
};

// @type SQLTypeInfo
// @brief a structure to capture all type information including
// length, precision, scale, etc.
template <template <class> class... TYPE_FACET_PACK>
class SQLTypeInfoCore : public TYPE_FACET_PACK<SQLTypeInfoCore<TYPE_FACET_PACK...> >... {
 public:
  SQLTypeInfoCore(SQLTypes t, int d, int s, bool n, EncodingType c, int p, SQLTypes st)
      : type(t)
      , subtype(st)
      , dimension(d)
      , scale(s)
      , notnull(n)
      , compression(c)
      , comp_param(p)
      , size(get_storage_size()) {}
  SQLTypeInfoCore(SQLTypes t, int d, int s, bool n)
      : type(t)
      , subtype(kNULLT)
      , dimension(d)
      , scale(s)
      , notnull(n)
      , compression(kENCODING_NONE)
      , comp_param(0)
      , size(get_storage_size()) {}
  SQLTypeInfoCore(SQLTypes t, int d, int s) : SQLTypeInfoCore(t, d, s, false) {}
  SQLTypeInfoCore(SQLTypes t, bool n)
      : type(t)
      , subtype(kNULLT)
      , dimension(0)
      , scale(0)
      , notnull(n)
      , compression(kENCODING_NONE)
      , comp_param(0)
      , size(get_storage_size()) {}
  SQLTypeInfoCore(SQLTypes t) : SQLTypeInfoCore(t, false) {}
  SQLTypeInfoCore(SQLTypes t, bool n, EncodingType c)
      : type(t)
      , subtype(kNULLT)
      , dimension(0)
      , scale(0)
      , notnull(n)
      , compression(c)
      , comp_param(0)
      , size(get_storage_size()) {}
  SQLTypeInfoCore()
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
  inline int get_logical_size() const {
    if (compression == kENCODING_FIXED || compression == kENCODING_DATE_IN_DAYS) {
      SQLTypeInfoCore ti(type, dimension, scale, notnull, kENCODING_NONE, 0, subtype);
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
  inline void set_type(SQLTypes t) { type = t; }
  inline void set_subtype(SQLTypes st) { subtype = st; }
  inline void set_dimension(int d) { dimension = d; }
  inline void set_precision(int d) { dimension = d; }
  inline void set_input_srid(int d) { dimension = d; }
  inline void set_scale(int s) { scale = s; }
  inline void set_output_srid(int s) { scale = s; }
  inline void set_notnull(bool n) { notnull = n; }
  inline void set_size(int s) { size = s; }
  inline void set_fixed_size() { size = get_storage_size(); }
  inline void set_compression(EncodingType c) { compression = c; }
  inline void set_comp_param(int p) { comp_param = p; }
#ifndef __CUDACC__
  inline std::string get_type_name() const {
    if (IS_GEO(type)) {
      std::string srid_string = "";
      if (get_output_srid() > 0)
        srid_string = ", " + std::to_string(get_output_srid());
      return type_name[(int)subtype] + "(" + type_name[(int)type] + srid_string + ")";
    }
    std::string ps = "";
    if (type == kDECIMAL || type == kNUMERIC || subtype == kDECIMAL ||
        subtype == kNUMERIC) {
      ps = "(" + std::to_string(dimension) + "," + std::to_string(scale) + ")";
    } else if (type == kTIMESTAMP) {
      ps = "(" + std::to_string(dimension) + ")";
    }
    if (type == kARRAY) {
      auto num_elems =
          (size > 0) ? std::to_string(size / (SQLTypeInfoCore(subtype, true).get_size()))
                     : "";
      return type_name[(int)subtype] + ps + "[" + num_elems + "]";
    }
    return type_name[(int)type] + ps;
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
  inline bool is_varlen_array() const { return type == kARRAY && size <= 0; }
  inline bool is_fixlen_array() const { return type == kARRAY && size > 0; }
  inline bool is_timeinterval() const {
    return type == kINTERVAL_DAY_TIME || type == kINTERVAL_YEAR_MONTH;
  }
  inline bool is_geometry() const { return IS_GEO(type); }

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

  HOST DEVICE inline bool operator!=(const SQLTypeInfoCore& rhs) const {
    return type != rhs.get_type() || subtype != rhs.get_subtype() ||
           dimension != rhs.get_dimension() || scale != rhs.get_scale() ||
           compression != rhs.get_compression() ||
           (compression != kENCODING_NONE && comp_param != rhs.get_comp_param() &&
            comp_param != TRANSIENT_DICT(rhs.get_comp_param())) ||
           notnull != rhs.get_notnull();
  }
  HOST DEVICE inline bool operator==(const SQLTypeInfoCore& rhs) const {
    return type == rhs.get_type() && subtype == rhs.get_subtype() &&
           dimension == rhs.get_dimension() && scale == rhs.get_scale() &&
           compression == rhs.get_compression() &&
           (compression == kENCODING_NONE || comp_param == rhs.get_comp_param() ||
            comp_param == TRANSIENT_DICT(rhs.get_comp_param())) &&
           notnull == rhs.get_notnull();
  }

  // FIX-ME:  Work through variadic base classes
  HOST DEVICE inline SQLTypeInfoCore& operator=(const SQLTypeInfoCore& rhs) {
    type = rhs.get_type();
    subtype = rhs.get_subtype();
    dimension = rhs.get_dimension();
    scale = rhs.get_scale();
    notnull = rhs.get_notnull();
    compression = rhs.get_compression();
    comp_param = rhs.get_comp_param();
    size = rhs.get_size();
    return (*this);
  }

  inline bool is_castable(const SQLTypeInfoCore& new_type_info) const {
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
        if (compression == kENCODING_DATE_IN_DAYS) {
          return d.timeval == NULL_INT;
        }
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
  inline SQLTypeInfoCore get_elem_type() const {
    return SQLTypeInfoCore(
        subtype, dimension, scale, notnull, compression, comp_param, kNULLT);
  }
  inline SQLTypeInfoCore get_array_type() const {
    return SQLTypeInfoCore(
        kARRAY, dimension, scale, notnull, compression, comp_param, type);
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
            return sizeof(time_t);
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
                return sizeof(int32_t);
              case 16:
                return 2;
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
        if (compression == kENCODING_DICT)
          return sizeof(int32_t);  // @TODO(wei) must check DictDescriptor
        break;
      case kARRAY:
        // TODO: return size for fixlen arrays?
        break;
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        break;
      default:
        break;
    }
    return -1;
  }
};

#ifndef __CUDACC__
// todo:  Get rid of preprocessor definition and move into Cuda Type Concept
template <template <class> class... TYPE_FACET_PACK>
std::string SQLTypeInfoCore<TYPE_FACET_PACK...>::type_name[kSQLTYPE_LAST] = {
    "NULL",
    "BOOLEAN",
    "CHAR",
    "VARCHAR",
    "NUMERIC",
    "DECIMAL",
    "INTEGER",
    "SMALLINT",
    "FLOAT",
    "DOUBLE",
    "TIME",
    "TIMESTAMP",
    "BIGINT",
    "TEXT",
    "DATE",
    "ARRAY",
    "INTERVAL_DAY_TIME",
    "INTERVAL_YEAR_MONTH",
    "POINT",
    "LINESTRING",
    "POLYGON",
    "MULTIPOLYGON",
    "TINYINT",
    "GEOMETRY",
    "GEOGRAPHY"};

template <template <class> class... TYPE_FACET_PACK>
std::string SQLTypeInfoCore<TYPE_FACET_PACK...>::comp_name[kENCODING_LAST] =
    {"NONE", "FIXED", "RL", "DIFF", "DICT", "SPARSE", "COMPRESSED", "DAYS"};
#endif

using SQLTypeInfo =
    SQLTypeInfoCore<ArrayContextTypeSizer, ExecutorTypePackaging, DateTimeFacilities>;

SQLTypes decimal_to_int_type(const SQLTypeInfo&);

#ifndef __CUDACC__
Datum StringToDatum(const std::string& s, SQLTypeInfo& ti);
std::string DatumToString(Datum d, const SQLTypeInfo& ti);
bool DatumEqual(const Datum, const Datum, const SQLTypeInfo& ti);
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info);
#endif

#include "../QueryEngine/DateAdd.h"
#include "../QueryEngine/DateTruncate.h"
#include "../QueryEngine/ExtractFromTime.h"

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

template <class T>
inline int64_t inline_int_null_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::min()
                                  : std::numeric_limits<T>::max();
}

template <class T>
inline int64_t max_valid_int_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::max()
                                  : std::numeric_limits<T>::max() - 1;
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
