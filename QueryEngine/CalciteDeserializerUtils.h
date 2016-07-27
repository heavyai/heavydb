#ifndef QUERYENGINE_CALCITEDESERIALIZERUTILS_H
#define QUERYENGINE_CALCITEDESERIALIZERUTILS_H

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
  CHECK(false);
  return kCOUNT;
}

inline SQLTypes to_sql_type(const std::string& type_name) {
  if (type_name == std::string("BIGINT")) {
    return kBIGINT;
  }
  if (type_name == std::string("INTEGER")) {
    return kINT;
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
  return kNULLT;
}

namespace Analyzer {

class Expr;

}  // Analyzer

SQLTypeInfo get_agg_type(const SQLAgg agg_kind, const Analyzer::Expr* arg_expr);

ExtractField to_datepart_field(const std::string&);

DatetruncField to_datediff_field(const std::string&);

#endif  // QUERYENGINE_CALCITEDESERIALIZERUTILS_H
