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

#ifndef QUERYENGINE_CALCITEDESERIALIZERUTILS_H
#define QUERYENGINE_CALCITEDESERIALIZERUTILS_H

#include "DateAdd.h"
#include "DateTruncate.h"

#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"
#include "Logger/Logger.h"

inline SQLOps to_sql_op(const std::string& op_str) {
  static const std::unordered_map<std::string, SQLOps> op_map = {
      {">", SQLOps::kGT},
      {"IS NOT DISTINCT FROM", SQLOps::kBW_EQ},
      {">=", SQLOps::kGE},
      {"<", SQLOps::kLT},
      {"<=", SQLOps::kLE},
      {"=", SQLOps::kEQ},
      {"<>", SQLOps::kNE},
      {"+", SQLOps::kPLUS},
      {"-", SQLOps::kMINUS},
      {"*", SQLOps::kMULTIPLY},
      {"/", SQLOps::kDIVIDE},
      {"MOD", SQLOps::kMODULO},
      {"AND", SQLOps::kAND},
      {"OR", SQLOps::kOR},
      {"CAST", SQLOps::kCAST},
      {"ENCODE_TEXT", SQLOps::kENCODE_TEXT},
      {"NOT", SQLOps::kNOT},
      {"IS NULL", SQLOps::kISNULL},
      {"IS NOT NULL", SQLOps::kISNOTNULL},
      {"PG_UNNEST", SQLOps::kUNNEST},
      {"IN", SQLOps::kIN}};

  auto it = op_map.find(op_str);
  if (it != op_map.end()) {
    return it->second;
  } else if (op_str == "PG_ANY" || op_str == "PG_ALL") {
    throw std::runtime_error("Invalid use of " + op_str + " operator");
  }
  return SQLOps::kFUNCTION;
}

inline SQLAgg to_agg_kind(const std::string& agg_name) {
  static const std::unordered_map<std::string, SQLAgg> agg_map = {
      {"COUNT", SQLAgg::kCOUNT},
      {"MIN", SQLAgg::kMIN},
      {"MAX", SQLAgg::kMAX},
      {"SUM", SQLAgg::kSUM},
      {"$SUM0", SQLAgg::kSUM},
      {"AVG", SQLAgg::kAVG},
      {"APPROX_COUNT_DISTINCT", SQLAgg::kAPPROX_COUNT_DISTINCT},
      {"APPROX_MEDIAN", SQLAgg::kAPPROX_QUANTILE},
      {"APPROX_PERCENTILE", SQLAgg::kAPPROX_QUANTILE},
      {"APPROX_QUANTILE", SQLAgg::kAPPROX_QUANTILE},
      {"ANY_VALUE", SQLAgg::kSAMPLE},
      {"SAMPLE", SQLAgg::kSAMPLE},
      {"LAST_SAMPLE", SQLAgg::kSAMPLE},
      {"SINGLE_VALUE", SQLAgg::kSINGLE_VALUE},
      {"MODE", SQLAgg::kMODE},
      {"COUNT_IF", SQLAgg::kCOUNT_IF},
      {"SUM_IF", SQLAgg::kSUM_IF}};

  auto it = agg_map.find(agg_name);
  if (it != agg_map.end()) {
    if (agg_name.compare("$SUM0") == 0) {
      VLOG(1)
          << "Convert $SUM0 aggregate function to an equivalent SUM aggregate function";
    }
    return it->second;
  } else {
    throw std::runtime_error("Aggregate function " + agg_name + " not supported");
  }
}

inline SQLTypes to_sql_type(const std::string& type_name) {
  static const std::unordered_map<std::string, SQLTypes> type_map = {
      {"BIGINT", SQLTypes::kBIGINT},
      {"INTEGER", SQLTypes::kINT},
      {"TINYINT", SQLTypes::kTINYINT},
      {"SMALLINT", SQLTypes::kSMALLINT},
      {"FLOAT", SQLTypes::kFLOAT},
      {"REAL", SQLTypes::kFLOAT},  // Same as FLOAT
      {"DOUBLE", SQLTypes::kDOUBLE},
      {"DECIMAL", SQLTypes::kDECIMAL},
      {"CHAR", SQLTypes::kTEXT},
      {"VARCHAR", SQLTypes::kTEXT},
      {"SYMBOL", SQLTypes::kTEXT},
      {"BOOLEAN", SQLTypes::kBOOLEAN},
      {"TIMESTAMP", SQLTypes::kTIMESTAMP},
      {"DATE", SQLTypes::kDATE},
      {"TIME", SQLTypes::kTIME},
      {"NULL", SQLTypes::kNULLT},
      {"ARRAY", SQLTypes::kARRAY},
      {"INTERVAL_DAY", SQLTypes::kINTERVAL_DAY_TIME},
      {"INTERVAL_HOUR", SQLTypes::kINTERVAL_DAY_TIME},
      {"INTERVAL_MINUTE", SQLTypes::kINTERVAL_DAY_TIME},
      {"INTERVAL_SECOND", SQLTypes::kINTERVAL_DAY_TIME},
      {"INTERVAL_MONTH", SQLTypes::kINTERVAL_YEAR_MONTH},
      {"INTERVAL_YEAR", SQLTypes::kINTERVAL_YEAR_MONTH},
      {"ANY", SQLTypes::kEVAL_CONTEXT_TYPE},
      {"TEXT", SQLTypes::kTEXT},
      {"POINT", SQLTypes::kPOINT},
      {"MULTIPOINT", SQLTypes::kMULTIPOINT},
      {"LINESTRING", SQLTypes::kLINESTRING},
      {"MULTILINESTRING", SQLTypes::kMULTILINESTRING},
      {"POLYGON", SQLTypes::kPOLYGON},
      {"MULTIPOLYGON", SQLTypes::kMULTIPOLYGON},
      {"GEOMETRY", SQLTypes::kGEOMETRY},
      {"GEOGRAPHY", SQLTypes::kGEOGRAPHY}};

  auto it = type_map.find(type_name);
  if (it != type_map.end()) {
    return it->second;
  } else {
    throw std::runtime_error("Unsupported type: " + type_name);
  }
}

namespace Analyzer {

class Constant;
class Expr;

}  // namespace Analyzer

SQLTypeInfo get_agg_type(const SQLAgg agg_kind, const Analyzer::Expr* arg_expr);

ExtractField to_datepart_field(const std::string&);

DateaddField to_dateadd_field(const std::string&);

DatetruncField to_datediff_field(const std::string&);

std::shared_ptr<Analyzer::Constant> make_fp_constant(const int64_t val,
                                                     const SQLTypeInfo& ti);

#endif  // QUERYENGINE_CALCITEDESERIALIZERUTILS_H
