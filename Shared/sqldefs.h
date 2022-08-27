/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @brief		Common Enum definitions for SQL processing.
 *
 */

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
  kOVERLAPS,
  kENCODE_TEXT,
  kINVALID_OP
};

#define IS_COMPARISON(X)                                                          \
  ((X) == kEQ || (X) == kBW_EQ || (X) == kOVERLAPS || (X) == kNE || (X) == kLT || \
   (X) == kGT || (X) == kLE || (X) == kGE)
#define IS_LOGIC(X) ((X) == kAND || (X) == kOR)
#define IS_ARITHMETIC(X) \
  ((X) == kMINUS || (X) == kPLUS || (X) == kMULTIPLY || (X) == kDIVIDE || (X) == kMODULO)
#define COMMUTE_COMPARISON(X) \
  ((X) == kLT ? kGT : (X) == kLE ? kGE : (X) == kGT ? kLT : (X) == kGE ? kLE : (X))
#define IS_UNARY(X)                                                                     \
  ((X) == kNOT || (X) == kUMINUS || (X) == kISNULL || (X) == kEXISTS || (X) == kCAST || \
   (X) == kENCODE_TEXT)
#define IS_EQUIVALENCE(X) ((X) == kEQ || (X) == kBW_EQ || (X) == kOVERLAPS)

enum SQLQualifier { kONE, kANY, kALL };

enum SQLAgg {
  kAVG,
  kMIN,
  kMAX,
  kSUM,
  kCOUNT,
  kAPPROX_COUNT_DISTINCT,
  kAPPROX_QUANTILE,
  kSAMPLE,
  kSINGLE_VALUE,
  kMODE,
  kINVALID_AGG
};

enum class SqlStringOpKind {
  /* Unary */
  LOWER = 1,
  UPPER,
  INITCAP,
  REVERSE,
  /* Binary */
  REPEAT,
  CONCAT,
  RCONCAT,
  /* Ternary */
  LPAD,
  RPAD,
  TRIM,
  LTRIM,
  RTRIM,
  SUBSTRING,
  OVERLAY,
  REPLACE,
  SPLIT_PART,
  /* 6 args */
  REGEXP_REPLACE,
  REGEXP_SUBSTR,
  JSON_VALUE,
  BASE64_ENCODE,
  BASE64_DECODE,
  TRY_STRING_CAST,  // string-to-numeric
  POSITION,         // string-to-numeric
  INVALID
};

enum class SqlWindowFunctionKind {
  // set MIN's enum val as one, and we use window function kind's enum vals
  // to classify a behavior of our runtime code for window framing
  // i.e., aggregate_##value_type##_values functions
  // todo (yoonmin): support FIRST_EXPR, LAST_EXPR, AND NTH_EXPR with framing
  MIN = 1,
  MAX,
  AVG,
  SUM,
  COUNT,
  FIRST_EXPR,
  LAST_EXPR,
  NTH_EXPR,
  ROW_NUMBER,
  RANK,
  DENSE_RANK,
  PERCENT_RANK,
  CUME_DIST,
  NTILE,
  LAG,
  LAG_IN_FRAME,
  LEAD,
  LEAD_IN_FRAME,
  FIRST_VALUE,
  LAST_VALUE,
  SUM_INTERNAL,  // For deserialization from Calcite only. Gets rewritten to a regular
                 // SUM.
  INVALID
};

enum class SqlWindowFrameBoundType {
  UNBOUNDED_PRECEDING = 1,
  EXPR_PRECEDING,
  CURRENT_ROW,
  EXPR_FOLLOWING,
  UNBOUNDED_FOLLOWING,
  UNKNOWN
};

enum SQLStmtType { kSELECT, kUPDATE, kINSERT, kDELETE, kCREATE_TABLE };

enum StorageOption { kDISK = 0, kGPU = 1, kCPU = 2 };

enum ViewRefreshOption { kMANUAL = 0, kAUTO = 1, kIMMEDIATE = 2 };

enum class JoinType { INNER, LEFT, SEMI, ANTI, INVALID };

#if !(defined(__CUDACC__) || defined(NO_BOOST))

#include <sstream>
#include <string>
#include "Logger/Logger.h"

inline std::string toString(const JoinType& join_type) {
  switch (join_type) {
    case JoinType::INNER:
      return "INNER";
    case JoinType::LEFT:
      return "LEFT";
    case JoinType::SEMI:
      return "SEMI";
    case JoinType::ANTI:
      return "ANTI";
    default:
      return "INVALID";
  }
}

inline std::string toString(const SQLQualifier& qualifier) {
  switch (qualifier) {
    case kONE:
      return "ONE";
    case kANY:
      return "ANY";
    case kALL:
      return "ALL";
  }
  LOG(FATAL) << "Invalid SQLQualifier: " << qualifier;
  return "";
}

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
    case kAPPROX_QUANTILE:
      return "APPROX_PERCENTILE";
    case kSAMPLE:
      return "SAMPLE";
    case kSINGLE_VALUE:
      return "SINGLE_VALUE";
    case kMODE:
      return "MODE";
    case kINVALID_AGG:
      return "INVALID";
    default:
      UNREACHABLE() << "Invalid aggregate kind: " << kind;
      return {};
  }
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
    case kENCODE_TEXT:
      return "ENCODE_TEXT";
    case kINVALID_OP:
      return "INVALID_OP";
  }
  LOG(FATAL) << "Invalid operation kind: " << op;
  return "";
}

inline std::ostream& operator<<(std::ostream& os, const SqlStringOpKind kind) {
  switch (kind) {
    case SqlStringOpKind::LOWER:
      return os << "LOWER";
    case SqlStringOpKind::UPPER:
      return os << "UPPER";
    case SqlStringOpKind::INITCAP:
      return os << "INITCAP";
    case SqlStringOpKind::REVERSE:
      return os << "REVERSE";
    case SqlStringOpKind::REPEAT:
      return os << "REPEAT";
    case SqlStringOpKind::CONCAT:
      return os << "CONCAT";
    case SqlStringOpKind::RCONCAT:
      return os << "RCONCAT";
    case SqlStringOpKind::LPAD:
      return os << "LPAD";
    case SqlStringOpKind::RPAD:
      return os << "RPAD";
    case SqlStringOpKind::TRIM:
      return os << "TRIM";
    case SqlStringOpKind::LTRIM:
      return os << "LTRIM";
    case SqlStringOpKind::RTRIM:
      return os << "RTRIM";
    case SqlStringOpKind::SUBSTRING:
      return os << "SUBSTRING";
    case SqlStringOpKind::OVERLAY:
      return os << "OVERLAY";
    case SqlStringOpKind::REPLACE:
      return os << "REPLACE";
    case SqlStringOpKind::SPLIT_PART:
      return os << "SPLIT_PART";
    case SqlStringOpKind::REGEXP_REPLACE:
      return os << "REGEXP_REPLACE";
    case SqlStringOpKind::REGEXP_SUBSTR:
      return os << "REGEXP_SUBSTR";
    case SqlStringOpKind::JSON_VALUE:
      return os << "JSON_VALUE";
    case SqlStringOpKind::BASE64_ENCODE:
      return os << "BASE64_ENCODE";
    case SqlStringOpKind::BASE64_DECODE:
      return os << "BASE64_DECODE";
    case SqlStringOpKind::TRY_STRING_CAST:
      return os << "TRY_STRING_CAST";
    case SqlStringOpKind::POSITION:
      return os << "POSITION";
    case SqlStringOpKind::INVALID:
      return os << "INVALID";
  }
  LOG(FATAL) << "Invalid string operation";
  // Make compiler happy
  return os << "INVALID";
}

inline SqlStringOpKind name_to_string_op_kind(const std::string& func_name) {
  if (func_name == "LOWER") {
    return SqlStringOpKind::LOWER;
  }
  if (func_name == "UPPER") {
    return SqlStringOpKind::UPPER;
  }
  if (func_name == "INITCAP") {
    return SqlStringOpKind::INITCAP;
  }
  if (func_name == "REVERSE") {
    return SqlStringOpKind::REVERSE;
  }
  if (func_name == "REPEAT") {
    return SqlStringOpKind::REPEAT;
  }
  if (func_name == "||") {
    return SqlStringOpKind::CONCAT;
  }
  if (func_name == "LPAD") {
    return SqlStringOpKind::LPAD;
  }
  if (func_name == "RPAD") {
    return SqlStringOpKind::RPAD;
  }
  if (func_name == "TRIM") {
    return SqlStringOpKind::TRIM;
  }
  if (func_name == "LTRIM") {
    return SqlStringOpKind::LTRIM;
  }
  if (func_name == "RTRIM") {
    return SqlStringOpKind::RTRIM;
  }
  if (func_name == "SUBSTRING") {
    return SqlStringOpKind::SUBSTRING;
  }
  if (func_name == "OVERLAY") {
    return SqlStringOpKind::OVERLAY;
  }
  if (func_name == "REPLACE") {
    return SqlStringOpKind::REPLACE;
  }
  if (func_name == "SPLIT_PART") {
    return SqlStringOpKind::SPLIT_PART;
  }
  if (func_name == "REGEXP_REPLACE") {
    return SqlStringOpKind::REGEXP_REPLACE;
  }
  if (func_name == "REGEXP_SUBSTR") {
    return SqlStringOpKind::REGEXP_SUBSTR;
  }
  if (func_name == "REGEXP_MATCH") {
    return SqlStringOpKind::REGEXP_SUBSTR;
  }
  if (func_name == "JSON_VALUE") {
    return SqlStringOpKind::JSON_VALUE;
  }
  if (func_name == "BASE64_ENCODE") {
    return SqlStringOpKind::BASE64_ENCODE;
  }
  if (func_name == "BASE64_DECODE") {
    return SqlStringOpKind::BASE64_DECODE;
  }
  if (func_name == "TRY_CAST") {
    return SqlStringOpKind::TRY_STRING_CAST;
  }
  if (func_name == "POSITION") {
    return SqlStringOpKind::POSITION;
  }
  LOG(FATAL) << "Invalid string function " << func_name << ".";
  return SqlStringOpKind::INVALID;
}

inline bool string_op_returns_string(const SqlStringOpKind kind) {
  switch (kind) {
    case SqlStringOpKind::TRY_STRING_CAST:
    case SqlStringOpKind::POSITION:
      return false;
    default:
      return true;
  }
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
    case SqlWindowFunctionKind::FIRST_EXPR:
      return "FIRST_EXPR";
    case SqlWindowFunctionKind::LAST_EXPR:
      return "LAST_EXPR";
    case SqlWindowFunctionKind::NTH_EXPR:
      return "NTH_EXPR";
    case SqlWindowFunctionKind::LEAD_IN_FRAME:
      return "LEAD_IN_FRAME";
    case SqlWindowFunctionKind::LAG_IN_FRAME:
      return "LAG_IN_FRAME";
    case SqlWindowFunctionKind::INVALID:
      return "INVALID";
  }
  LOG(FATAL) << "Invalid window function kind.";
  return "";
}

inline std::string toString(const SqlWindowFrameBoundType& kind) {
  switch (kind) {
    case SqlWindowFrameBoundType::UNBOUNDED_PRECEDING:
      return "UNBOUNDED_PRECEDING";
    case SqlWindowFrameBoundType::EXPR_PRECEDING:
      return "EXPR_PRECEDING";
    case SqlWindowFrameBoundType::CURRENT_ROW:
      return "CURRENT_ROW";
    case SqlWindowFrameBoundType::EXPR_FOLLOWING:
      return "EXPR_FOLLOWING";
    case SqlWindowFrameBoundType::UNBOUNDED_FOLLOWING:
      return "UNBOUNDED_FOLLOWING";
    case SqlWindowFrameBoundType::UNKNOWN:
      return "UNKNOWN";
  }
  LOG(FATAL) << "Invalid window function bound type.";
  return "";
}

#endif  // #if !(defined(__CUDACC__) || defined(NO_BOOST))

#endif  // SQLDEFS_H
