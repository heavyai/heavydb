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
	kTEXT = 13
};

#define IS_NUMBER(T) (((T) == kINT) || ((T) == kSMALLINT) || ((T) == kDOUBLE) || ((T) == kFLOAT) || ((T) == kBIGINT) || ((T) == kNUMERIC) || ((T) == kDECIMAL))
#define IS_STRING(T) (((T) == kTEXT) || ((T) == kVARCHAR) || ((T) == kCHAR))

typedef union {
	bool boolval;
	int16_t smallintval;
	int32_t intval;
	int64_t bigintval;
	float floatval;
	double doubleval;
	void *pointerval; // by reference values
} Datum;

// @type SQLTypeInfo
// @brief a structure to capture all type information including
// length, precision, scale, etc.
struct SQLTypeInfo {
	SQLTypes type = kNULLT; // type id
	int dimension = 0; // VARCHAR/CHAR length or NUMERIC/DECIMAL precision
	int scale = 0; // NUMERIC/DECIMAL scale
	bool notnull = false; // nullable?  a hint, not used for type checking

	bool operator!=(const SQLTypeInfo &rhs) const {
		return type != rhs.type || dimension != rhs.dimension || scale != rhs.scale;
	}
	bool operator==(const SQLTypeInfo &rhs) const {
		return type == rhs.type && dimension == rhs.dimension && scale == rhs.scale;
	}
	void operator=(const SQLTypeInfo &rhs) {
		type = rhs.type;
		dimension = rhs.dimension;
		scale = rhs.scale;
		notnull = rhs.notnull;
	}
};

#endif // SQLTYPES_H
