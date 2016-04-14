/**
 * @file		sqldefs.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Common Enum definitions for SQL processing.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef SQLDEFS_H
#define SQLDEFS_H

// must not change the order without keeping the array in OperExpr::to_string
// in sync.
enum SQLOps {
  kEQ = 0,
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
  kMODULO,
  kUMINUS,
  kISNULL,
  kISNOTNULL,
  kEXISTS,
  kCAST,
  kARRAY_AT,
  kUNNEST,
  kFUNCTION
};

#define IS_COMPARISON(X) ((X) == kEQ || (X) == kNE || (X) == kLT || (X) == kGT || (X) == kLE || (X) == kGE)
#define IS_LOGIC(X) ((X) == kAND || (X) == kOR)
#define IS_ARITHMETIC(X) ((X) == kMINUS || (X) == kPLUS || (X) == kMULTIPLY || (X) == kDIVIDE || (X) == kMODULO)
#define COMMUTE_COMPARISON(X) ((X) == kLT ? kGT : (X) == kLE ? kGE : (X) == kGT ? kLT : (X) == kGE ? kLE : (X))
#define IS_UNARY(X) ((X) == kNOT || (X) == kUMINUS || (X) == kISNULL || (X) == kEXISTS || (X) == kCAST)

enum SQLQualifier { kONE, kANY, kALL };

enum SQLAgg { kAVG, kMIN, kMAX, kSUM, kCOUNT };

enum SQLStmtType { kSELECT, kUPDATE, kINSERT, kDELETE, kCREATE_TABLE };

enum StorageOption { kDISK = 0, kGPU = 1, kCPU = 2 };

enum ViewRefreshOption { kMANUAL = 0, kAUTO = 1, kIMMEDIATE = 2 };

enum class JoinType { INNER, LEFT, INVALID };

#endif  // SQLDEFS_H
