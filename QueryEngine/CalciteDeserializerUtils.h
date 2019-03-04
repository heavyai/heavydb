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

#ifndef QUERYENGINE_CALCITEDESERIALIZERUTILS_H
#define QUERYENGINE_CALCITEDESERIALIZERUTILS_H

#include "DateAdd.h"
#include "DateTruncate.h"

#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"

#include <glog/logging.h>

inline SQLOps to_sql_op(const std::string& op_str) {
  if (op_str == std::string(">")) {
    return kGT;
  }
  if (op_str == std::string(">=")) {
    return kGE;
  }
  if (op_str == std::string("<")) {
    return kLT;
  }
  if (op_str == std::string("<=")) {
    return kLE;
  }
  if (op_str == std::string("=")) {
    return kEQ;
  }
  if (op_str == std::string("<>")) {
    return kNE;
  }
  if (op_str == std::string("+")) {
    return kPLUS;
  }
  if (op_str == std::string("-")) {
    return kMINUS;
  }
  if (op_str == std::string("*")) {
    return kMULTIPLY;
  }
  if (op_str == std::string("/")) {
    return kDIVIDE;
  }
  if (op_str == "MOD") {
    return kMODULO;
  }
  if (op_str == std::string("AND")) {
    return kAND;
  }
  if (op_str == std::string("OR")) {
    return kOR;
  }
  if (op_str == std::string("CAST")) {
    return kCAST;
  }
  if (op_str == std::string("NOT")) {
    return kNOT;
  }
  if (op_str == std::string("IS NULL")) {
    return kISNULL;
  }
  if (op_str == std::string("IS NOT NULL")) {
    return kISNOTNULL;
  }
  if (op_str == std::string("PG_UNNEST")) {
    return kUNNEST;
  }
  if (op_str == std::string("PG_ANY") || op_str == std::string("PG_ALL")) {
    throw std::runtime_error("Invalid use of " + op_str + " operator");
  }
  if (op_str == std::string("IN")) {
    return kIN;
  }
  return kFUNCTION;
}

inline SQLAgg to_agg_kind(const std::string& agg_name) {
  if (agg_name == std::string("COUNT")) {
    return kCOUNT;
  }
  if (agg_name == std::string("MIN")) {
    return kMIN;
  }
  if (agg_name == std::string("MAX")) {
    return kMAX;
  }
  if (agg_name == std::string("SUM")) {
    return kSUM;
  }
  if (agg_name == std::string("AVG")) {
    return kAVG;
  }
  if (agg_name == std::string("APPROX_COUNT_DISTINCT")) {
    return kAPPROX_COUNT_DISTINCT;
  }
  if (agg_name == std::string("SAMPLE") || agg_name == std::string("LAST_SAMPLE")) {
    return kSAMPLE;
  }
  throw std::runtime_error("Aggregate function " + agg_name + " not supported");
}

inline SQLTypes to_sql_type(const std::string& type_name) {
  if (type_name == std::string("BIGINT")) {
    return kBIGINT;
  }
  if (type_name == std::string("INTEGER")) {
    return kINT;
  }
  if (type_name == std::string("TINYINT")) {
    return kTINYINT;
  }
  if (type_name == std::string("SMALLINT")) {
    return kSMALLINT;
  }
  if (type_name == std::string("FLOAT")) {
    return kFLOAT;
  }
  if (type_name == std::string("DOUBLE")) {
    return kDOUBLE;
  }
  if (type_name == std::string("DECIMAL")) {
    return kDECIMAL;
  }
  if (type_name == std::string("CHAR") || type_name == std::string("VARCHAR")) {
    return kTEXT;
  }
  if (type_name == std::string("BOOLEAN")) {
    return kBOOLEAN;
  }
  if (type_name == std::string("TIMESTAMP")) {
    return kTIMESTAMP;
  }
  if (type_name == std::string("DATE")) {
    return kDATE;
  }
  if (type_name == std::string("TIME")) {
    return kTIME;
  }
  if (type_name == std::string("NULL")) {
    return kNULLT;
  }
  if (type_name == std::string("ARRAY")) {
    return kARRAY;
  }
  if (type_name == std::string("INTERVAL_DAY") ||
      type_name == std::string("INTERVAL_HOUR") ||
      type_name == std::string("INTERVAL_MINUTE") ||
      type_name == std::string("INTERVAL_SECOND")) {
    return kINTERVAL_DAY_TIME;
  }
  if (type_name == std::string("INTERVAL_MONTH") ||
      type_name == std::string("INTERVAL_YEAR")) {
    return kINTERVAL_YEAR_MONTH;
  }
  if (type_name == std::string("ANY")) {
    return kEVAL_CONTEXT_TYPE;
  }

  throw std::runtime_error("Unsupported type: " + type_name);
}

namespace Analyzer {

class Constant;
class Expr;

}  // namespace Analyzer

SQLTypeInfo get_agg_type(const SQLAgg agg_kind, const Analyzer::Expr* arg_expr);

ExtractField to_datepart_field(const std::string&);

DateaddField to_dateadd_field(const std::string&);

DatetruncField to_datediff_field(const std::string&);

ExtractField to_extract_field(const std::string&);

DatetruncField to_datetrunc_field(const std::string&);

std::shared_ptr<Analyzer::Constant> make_fp_constant(const int64_t val,
                                                     const SQLTypeInfo& ti);

#endif  // QUERYENGINE_CALCITEDESERIALIZERUTILS_H
