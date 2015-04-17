/**
 * @file		sqltypes.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Constants for Builtin SQL Types supported by MapD
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef SQLTYPES_H
#define SQLTYPES_H

#include <stdint.h>
#include <ctime>
#include <cfloat>
#include <string>
#include <vector>
#include <cassert>

// must not change because these values persist in catalogs.
enum SQLTypes {
	kNULLT = 0, // type for null values
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
  kSQLTYPE_LAST = 15
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
#ifndef __CUDACC__
	std::string *stringval; // string value
#endif
} Datum;

#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

struct VarlenDatum {
	int length;
	int8_t *pointer;
	bool is_null;

	DEVICE VarlenDatum() : length(0), pointer(NULL), is_null(true) {}
	VarlenDatum(int l, int8_t *p, bool n) : length(l), pointer(p), is_null(n) {}
};

#ifndef __CUDACC__
union DataBlockPtr {
	int8_t *numbersPtr;
	std::vector<std::string> *stringsPtr;
  std::vector<std::vector<int8_t>> *tok8dictPtr; // single byte tokenized dictionary encoding array
  std::vector<std::vector<int16_t>> *tok16dictPtr; // double byte tokenized dictionary encoding array
  std::vector<std::vector<int32_t>> *tok32dictPtr; // 4-byte tokenized dictionary encoding array
};
#endif

// must not change because these values persist in catalogs.
enum EncodingType {
	kENCODING_NONE = 0, // no encoding
	kENCODING_FIXED = 1, // Fixed-bit encoding
	kENCODING_RL = 2, // Run Length encoding
	kENCODING_DIFF = 3, // Differential encoding
	kENCODING_DICT = 4, // Dictionary encoding
	kENCODING_SPARSE = 5, // Null encoding for sparse columns
	kENCODING_TOKDICT = 6, // Tokenized-Dictionary encoding
  kENCODING_LAST = 7
};

#define IS_INTEGER(T) (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kBIGINT))
#define IS_NUMBER(T) (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kDOUBLE) || ((T) == kFLOAT) || ((T) == kBIGINT) || ((T) == kNUMERIC) || ((T) == kDECIMAL))
#define IS_STRING(T) (((T) == kTEXT) || ((T) == kVARCHAR) || ((T) == kCHAR))
#define IS_TIME(T) (((T) == kTIME) || ((T) == kTIMESTAMP) || ((T) == kDATE))

#define NULL_BOOLEAN    INT8_MIN
#define NULL_TINYINT    INT8_MIN
#define NULL_SMALLINT   INT16_MIN
#define NULL_INT        INT32_MIN
#define NULL_BIGINT     INT64_MIN
#define NULL_FLOAT      FLT_MIN
#define NULL_DOUBLE     DBL_MIN


// @type SQLTypeInfo
// @brief a structure to capture all type information including
// length, precision, scale, etc.
class SQLTypeInfo {
  public:
    SQLTypeInfo(SQLTypes t, int d, int s, bool n, EncodingType c, int p, int es) : type(t), dimension(d), scale(s), notnull(n), compression(c), comp_param(p), size(get_storage_size()), elem_size(es) {}
    SQLTypeInfo(SQLTypes t, int d, int s, bool n) : type(t), dimension(d), scale(s), notnull(n), compression(kENCODING_NONE), comp_param(0), size(get_storage_size()), elem_size(0) {}
    explicit SQLTypeInfo(SQLTypes t) : type(t), dimension(0), scale(0), notnull(false), compression(kENCODING_NONE), comp_param(0), size(get_storage_size()), elem_size(0) {}
    SQLTypeInfo() : type(kNULLT), dimension(0), scale(0), notnull(false), compression(kENCODING_NONE), comp_param(0), size(0), elem_size(0) {}

    DEVICE inline SQLTypes get_type() const { return type; }
    inline int get_dimension() const { return dimension; }
    inline int get_precision() const { return dimension; }
    inline int get_scale() const { return scale; }
    inline bool get_notnull() const { return notnull; }
    DEVICE inline EncodingType get_compression() const { return compression; }
    DEVICE inline int get_comp_param() const { return comp_param; }
    inline int get_size() const { return size; }
    inline int get_elem_size() const { return elem_size; }
    inline void set_type(SQLTypes t) { type = t; }
    inline void set_dimension(int d) { dimension = d; }
    inline void set_precision(int d) { dimension = d; }
    inline void set_scale(int s) { scale = s; }
    inline void set_notnull(bool n) { notnull = n; }
    inline void set_size(int s) { size = s; }
    inline void set_fixed_size() { size = get_storage_size(); }
    inline void set_elem_size(int s) { elem_size = s; }
    inline void set_compression(EncodingType c) { compression = c; }
    inline void set_comp_param(int p) { comp_param = p; }
#ifndef __CUDACC__
    inline std::string get_type_name() const { return type_name[(int)type]; }
    inline std::string get_compression_name() const { return comp_name[(int)compression]; }
#endif
    inline bool is_string() const { return IS_STRING(type); }
    inline bool is_integer() const { return IS_INTEGER(type); }
    inline bool is_fp() const { return type == kFLOAT || type == kDOUBLE; }
    inline bool is_number() const { return IS_NUMBER(type); }
    inline bool is_time() const { return IS_TIME(type); }
    inline bool is_boolean() const { return type == kBOOLEAN; }

		inline bool is_varlen() const { return IS_STRING(type) && compression != kENCODING_DICT; }


    DEVICE inline bool operator!=(const SQLTypeInfo &rhs) const {
      return type != rhs.get_type() || dimension != rhs.get_dimension() || scale != rhs.get_scale() || compression != rhs.get_compression() || comp_param != rhs.get_comp_param();
    }
    DEVICE inline bool operator==(const SQLTypeInfo &rhs) const {
      return type == rhs.get_type() && dimension == rhs.get_dimension() && scale == rhs.get_scale() && compression == rhs.get_compression() && comp_param == rhs.get_comp_param();
    }
    DEVICE inline void operator=(const SQLTypeInfo &rhs) {
      type = rhs.get_type();
      dimension = rhs.get_dimension();
      scale = rhs.get_scale();
      notnull = rhs.get_notnull();
      compression = rhs.get_compression();
      comp_param = rhs.get_comp_param();
      size = rhs.get_size();
      elem_size = rhs.get_elem_size();
    }
    DEVICE inline bool is_castable(const SQLTypeInfo &new_type_info) const {
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
      else if (type == kBOOLEAN && new_type_info.is_number())
        return true;
      else
        return false;
    }
    DEVICE inline bool is_null(const Datum &d) const {
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
          if (sizeof(time_t) == 4)
            return d.timeval == NULL_INT;
          return d.timeval == NULL_BIGINT;
        case kTEXT:
        case kVARCHAR:
        case kCHAR:
          // @TODO handle null strings
          break;
        case kNULLT:
          return true;
        default:
          break;
      }
      return false;
    }
    DEVICE inline bool is_null(const int8_t *val) const {
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
  private:
    SQLTypes type; // type id
    int dimension; // VARCHAR/CHAR length or NUMERIC/DECIMAL precision
    int scale; // NUMERIC/DECIMAL scale
    bool notnull; // nullable?  a hint, not used for type checking
    EncodingType compression; // compression scheme
    int comp_param; // compression parameter when applicable for certain schemes
    int size; // size of the type in bytes.  -1 for variable size
    int elem_size; // size of array elements in bytes, 1, 2 or 4.  for internal use for tokenized dictionary encoding only.
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
							return comp_param/8;
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
							return comp_param/8;
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
							return comp_param/8;
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
						assert(false); // not supported yet
				case kDATE:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(time_t);
						case kENCODING_FIXED:
							return comp_param/8;
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
            return 0; // means unknown.  will have to be set from DictDescriptor
          break;
				default:
					break;
			}
			return -1;
		}
};

#ifndef __CUDACC__
Datum
StringToDatum(const std::string &s, SQLTypeInfo &ti);
std::string
DatumToString(Datum d, const SQLTypeInfo &ti);
#endif

#include "../QueryEngine/ExtractFromTime.h"

typedef int32_t StringOffsetT;

#endif // SQLTYPES_H
