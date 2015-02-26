/**
 * @file		sqltypes.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Constants for Builtin SQL Types supported by MapD
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef SQLTYPES_H
#define SQLTYPES_H

#include <cstdint>
#include <ctime>
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
	int16_t smallintval;
	int32_t intval;
	int64_t bigintval;
	std::time_t timeval;
	float floatval;
	double doubleval;
	std::string *stringval; // string value
} Datum;

struct VarlenDatum {
	int length;
	int8_t *pointer;
	bool is_null;

	VarlenDatum() : length(0), pointer(nullptr), is_null(true) {}
	VarlenDatum(int l, int8_t *p, bool n) : length(l), pointer(p), is_null(n) {}
};

union DataBlockPtr {
	int8_t *numbersPtr;
	std::vector<std::string> *stringsPtr;
};

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

// @type SQLTypeInfo
// @brief a structure to capture all type information including
// length, precision, scale, etc.
class SQLTypeInfo {
  public:
    SQLTypeInfo(SQLTypes t, int d, int s, bool n, EncodingType c, int p) : type(t), dimension(d), scale(s), notnull(n), compression(c), comp_param(p) {}
    SQLTypeInfo(SQLTypes t, int d, int s, bool n) : type(t), dimension(d), scale(s), notnull(n), compression(kENCODING_NONE), comp_param(0) {}
    SQLTypeInfo(SQLTypes t) : type(t), dimension(0), scale(0), notnull(false), compression(kENCODING_NONE), comp_param(0) {}
    SQLTypeInfo() : type(kNULLT), dimension(0), scale(0), notnull(false), compression(kENCODING_NONE), comp_param(0) {}

    inline SQLTypes get_type() const { return type; }
    inline int get_dimension() const { return dimension; }
    inline int get_precision() const { return dimension; }
    inline int get_scale() const { return scale; }
    inline bool get_notnull() const { return notnull; }
    inline EncodingType get_compression() const { return compression; }
    inline int get_comp_param() const { return comp_param; }
    inline void set_type(SQLTypes t) { type = t; }
    inline void set_dimension(int d) { dimension = d; }
    inline void set_precision(int d) { dimension = d; }
    inline void set_scale(int s) { scale = s; }
    inline void set_notnull(bool n) { notnull = n; }
    inline void set_compression(EncodingType c) { compression = c; }
    inline void set_comp_param(int p) { comp_param = p; }
    inline std::string get_type_name() const { return type_name[(int)type]; }
    inline std::string get_compression_name() const { return comp_name[(int)compression]; }
    inline bool is_string() const { return IS_STRING(type); }
    inline bool is_integer() const { return IS_INTEGER(type); }
    inline bool is_number() const { return IS_NUMBER(type); }
    inline bool is_time() const { return IS_TIME(type); }

		inline bool is_varlen() const { return IS_STRING(type) && compression != kENCODING_DICT; }

		inline int get_storage_size() const {
			switch (type) {
				case kBOOLEAN:
					return sizeof(int8_t);
				case kSMALLINT:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(int16_t);
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
				case kINT:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(int32_t);
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
				case kBIGINT:
				case kNUMERIC:
				case kDECIMAL:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(int64_t);
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
            return sizeof(int32_t);
          break;
				default:
					break;
			}
			return -1;
		}

    inline bool operator!=(const SQLTypeInfo &rhs) const {
      return type != rhs.get_type() || dimension != rhs.get_dimension() || scale != rhs.get_scale() || compression != rhs.get_compression() || comp_param != rhs.get_comp_param();
    }
    inline bool operator==(const SQLTypeInfo &rhs) const {
      return type == rhs.get_type() && dimension == rhs.get_dimension() && scale == rhs.get_scale() && compression == rhs.get_compression() && comp_param == rhs.get_comp_param();
    }
    inline void operator=(const SQLTypeInfo &rhs) {
      type = rhs.get_type();
      dimension = rhs.get_dimension();
      scale = rhs.get_scale();
      notnull = rhs.get_notnull();
      compression = rhs.get_compression();
      comp_param = rhs.get_comp_param();
    }
    inline bool is_castable(const SQLTypeInfo &new_type_info) const {
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
  private:
    SQLTypes type; // type id
    int dimension; // VARCHAR/CHAR length or NUMERIC/DECIMAL precision
    int scale; // NUMERIC/DECIMAL scale
    bool notnull; // nullable?  a hint, not used for type checking
    EncodingType compression; // compression scheme
    int comp_param; // compression parameter when applicable for certain schemes
    std::string type_name[kSQLTYPE_LAST] = { "NULL", "BOOLEAN", "CHAR", "VARCHAR", "NUMERIC", "DECIMAL", "INTEGER", "SMALLINT", "FLOAT", "DOUBLE", "TIME", "TIMESTAMP", "BIGINT", "TEXT", "DATE" };
    std::string comp_name[kENCODING_LAST] = { "NONE", "FIXED", "RL", "DIFF", "DICT", "SPARSE", "TOKEN_DICT" };
};

Datum
StringToDatum(const std::string &s, SQLTypeInfo &ti);
std::string
DatumToString(Datum d, const SQLTypeInfo &ti);

#include "../QueryEngine/ExtractFromTime.h"

#endif // SQLTYPES_H
