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
  kBW_EQ,
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
  kFUNCTION,
  kIN,
  kOVERLAPS
};

#define IS_COMPARISON(X)                                                          \
  ((X) == kEQ || (X) == kBW_EQ || (X) == kOVERLAPS || (X) == kNE || (X) == kLT || \
   (X) == kGT || (X) == kLE || (X) == kGE)
#define IS_LOGIC(X) ((X) == kAND || (X) == kOR)
#define IS_ARITHMETIC(X) \
  ((X) == kMINUS || (X) == kPLUS || (X) == kMULTIPLY || (X) == kDIVIDE || (X) == kMODULO)
#define COMMUTE_COMPARISON(X) \
  ((X) == kLT ? kGT : (X) == kLE ? kGE : (X) == kGT ? kLT : (X) == kGE ? kLE : (X))
#define IS_UNARY(X) \
  ((X) == kNOT || (X) == kUMINUS || (X) == kISNULL || (X) == kEXISTS || (X) == kCAST)
#define IS_EQUIVALENCE(X) ((X) == kEQ || (X) == kBW_EQ || (X) == kOVERLAPS)

enum SQLQualifier { kONE, kANY, kALL };

enum SQLAgg {
  kAVG,
  kMIN,
  kMAX,
  kSUM,
  kCOUNT,
  kAPPROX_COUNT_DISTINCT,
  kSAMPLE,
  kSINGLE_VALUE
};

enum class SqlWindowFunctionKind {
  ROW_NUMBER,
  RANK,
  DENSE_RANK,
  PERCENT_RANK,
  CUME_DIST,
  NTILE,
  LAG,
  LEAD,
  FIRST_VALUE,
  LAST_VALUE,
  AVG,
  MIN,
  MAX,
  SUM,
  COUNT,
  SUM_INTERNAL  // For deserialization from Calcite only. Gets rewritten to a regular SUM.
};

enum SQLStmtType { kSELECT, kUPDATE, kINSERT, kDELETE, kCREATE_TABLE };

enum StorageOption { kDISK = 0, kGPU = 1, kCPU = 2 };

enum ViewRefreshOption { kMANUAL = 0, kAUTO = 1, kIMMEDIATE = 2 };

enum class JoinType { INNER, LEFT, INVALID };

#ifndef __CUDACC__

#include <string>
#include "Logger/Logger.h"

inline std::string toString(const SQLAgg& kind) {
  switch (kind) {
    case kAVG:
      return "AVG";
    case kMIN:
      return "MIN";
    case kMAX:
      return "MAX";
    case kSUM:
      return "SUM";
    case kCOUNT:
      return "COUNT";
    case kAPPROX_COUNT_DISTINCT:
      return "APPROX_COUNT_DISTINCT";
    case kSAMPLE:
      return "SAMPLE";
    case kSINGLE_VALUE:
      return "SINGLE_VALUE";
  }
  LOG(FATAL) << "Invalid aggregate kind: " << kind;
  return "";
}

inline std::string toString(const SQLOps& op) {
  switch (op) {
    case kEQ:
      return "EQ";
    case kBW_EQ:
      return "BW_EQ";
    case kNE:
      return "NE";
    case kLT:
      return "LT";
    case kGT:
      return "GT";
    case kLE:
      return "LE";
    case kGE:
      return "GE";
    case kAND:
      return "AND";
    case kOR:
      return "OR";
    case kNOT:
      return "NOT";
    case kMINUS:
      return "MINUS";
    case kPLUS:
      return "PLUS";
    case kMULTIPLY:
      return "MULTIPLY";
    case kDIVIDE:
      return "DIVIDE";
    case kMODULO:
      return "MODULO";
    case kUMINUS:
      return "UMINUS";
    case kISNULL:
      return "ISNULL";
    case kISNOTNULL:
      return "ISNOTNULL";
    case kEXISTS:
      return "EXISTS";
    case kCAST:
      return "CAST";
    case kARRAY_AT:
      return "ARRAY_AT";
    case kUNNEST:
      return "UNNEST";
    case kFUNCTION:
      return "FUNCTION";
    case kIN:
      return "IN";
    case kOVERLAPS:
      return "OVERLAPS";
  }
  LOG(FATAL) << "Invalid operation kind: " << op;
  return "";
}

inline std::string toString(const SqlWindowFunctionKind& kind) {
  switch (kind) {
    case SqlWindowFunctionKind::ROW_NUMBER:
      return "ROW_NUMBER";
    case SqlWindowFunctionKind::RANK:
      return "RANK";
    case SqlWindowFunctionKind::DENSE_RANK:
      return "DENSE_RANK";
    case SqlWindowFunctionKind::PERCENT_RANK:
      return "PERCENT_RANK";
    case SqlWindowFunctionKind::CUME_DIST:
      return "CUME_DIST";
    case SqlWindowFunctionKind::NTILE:
      return "NTILE";
    case SqlWindowFunctionKind::LAG:
      return "LAG";
    case SqlWindowFunctionKind::LEAD:
      return "LEAD";
    case SqlWindowFunctionKind::FIRST_VALUE:
      return "FIRST_VALUE";
    case SqlWindowFunctionKind::LAST_VALUE:
      return "LAST_VALUE";
    case SqlWindowFunctionKind::AVG:
      return "AVG";
    case SqlWindowFunctionKind::MIN:
      return "MIN";
    case SqlWindowFunctionKind::MAX:
      return "MAX";
    case SqlWindowFunctionKind::SUM:
      return "SUM";
    case SqlWindowFunctionKind::COUNT:
      return "COUNT";
    case SqlWindowFunctionKind::SUM_INTERNAL:
      return "SUM_INTERNAL";
  }
  LOG(FATAL) << "Invalid window function kind.";
  return "";
}

#endif

#endif  // SQLDEFS_H
