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
	kNE,
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
	kEXISTS,
	kCAST
};

#define IS_COMPARISON(X) ((X) == kEQ || (X) == kNE || (X) == kLT || (X) == kGT || (X) == kLE || (X) == kGE)
#define IS_LOGIC(X) ((X) == kAND || (X) == kOR)
#define IS_ARITHMETIC(X) ((X) == kMINUS || (X) == kPLUS || (X) == kMULTIPLY || (X) == kDIVIDE)
#define COMMUTE_COMPARISON(X) ((X) == kLT ? kGT : (X) == kLE ? kGE : (X) == kGT ? kLT : (X) == kGE ? kLE : (X))

enum SQLQualifier {
	kONE,
	kANY,
	kALL
};

enum SQLAgg {
	kAVG,
	kMIN,
	kMAX,
	kSUM,
	kCOUNT
};

enum SQLStmtType {
	kSELECT,
	kUPDATE,
	kINSERT,
	kDELETE,
	kCREATE_TABLE
};

#endif // SQLDEFS_H
