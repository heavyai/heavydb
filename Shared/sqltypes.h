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
}

typedef union {
	/* by value datum
	bool boolval;
	int16_t smallintval;
	int32_t intval;
	int64_t bigintval;
	float floatval;
	double doubleval;
	/* by reference datum */
	void *pointerval;
} Datum;

#endif // SQLTYPES_H
