/**
 * @file		sqldefs.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Common Enum definitions for SQL processing.
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef SQLDEFS_H
#define SQLDEFS_H

enum SQLOps {
	kEQ,
	kNEQ,
	kLT,
	kGT,
	kLE,
	kGE,
	kAND,
	kOR,
	kNOT,
	kMINUS,
	kPLUS,
	kMULTIPLY,
	kDIVIDE,
	kUMINUS,
	kISNULL,
	kEXISTS
}

enum SQLQualifer {
	kONE,
	kANY,
	kALL
}

enum SQLAgg {
	kAVG,
	kMIN,
	kMAX,
	kSUM,
	kCOUNT
}

enum SQLStmtType {
	kSELECT,
	kUPDATE,
	kINSERT,
	kDELETE,
	kCREATE_TABLE
}

#endif // SQLDEFS_H
