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

#include "CalciteDeserializerUtils.h"

#include "../Analyzer/Analyzer.h"
#include "Logger/Logger.h"

#include <boost/algorithm/string.hpp>

extern bool g_bigint_count;

SQLTypeInfo get_agg_type(const SQLAgg agg_kind, const Analyzer::Expr* arg_expr) {
  switch (agg_kind) {
    case kCOUNT:
      return SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false);
    case kMIN:
    case kMAX:
      return arg_expr->get_type_info();
    case kSUM:
      return arg_expr->get_type_info().is_integer() ? SQLTypeInfo(kBIGINT, false)
                                                    : arg_expr->get_type_info();
    case kAVG:
      return SQLTypeInfo(kDOUBLE, false);
    case kAPPROX_COUNT_DISTINCT:
      return SQLTypeInfo(kBIGINT, false);
    case kAPPROX_QUANTILE:
      return SQLTypeInfo(kDOUBLE, false);
    case kSINGLE_VALUE:
      if (arg_expr->get_type_info().is_varlen()) {
        throw std::runtime_error("SINGLE_VALUE not supported on '" +
                                 arg_expr->get_type_info().get_type_name() + "' input.");
      }  // else fall through
    case kSAMPLE:
    case kMODE:
      return arg_expr->get_type_info();
    default:
      UNREACHABLE() << "Unsupported agg_kind: " << agg_kind;
      return {};
  }
}

ExtractField to_datepart_field(const std::string& field) {
  ExtractField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = kYEAR;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = kQUARTER;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = kMONTH;
  } else if (boost::iequals(field, "dayofyear") || boost::iequals(field, "dy") ||
             boost::iequals(field, "y")) {
    fieldno = kDOY;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = kDAY;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = kWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = kWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = kWEEK_SATURDAY;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = kHOUR;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = kMINUTE;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = kSECOND;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = kMILLISECOND;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = kMICROSECOND;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = kNANOSECOND;
  } else if (boost::iequals(field, "weekday") || boost::iequals(field, "dw")) {
    fieldno = kISODOW;
  } else if (boost::iequals(field, "quarterday") || boost::iequals(field, "dq")) {
    fieldno = kQUARTERDAY;
  } else {
    throw std::runtime_error("Unsupported field in DATEPART function: " + field);
  }
  return fieldno;
}

DateaddField to_dateadd_field(const std::string& field) {
  DateaddField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = daYEAR;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = daQUARTER;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = daMONTH;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = daDAY;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = daWEEK;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = daHOUR;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = daMINUTE;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = daSECOND;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = daMILLISECOND;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = daMICROSECOND;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = daNANOSECOND;
  } else if (boost::iequals(field, "weekday") || boost::iequals(field, "dw")) {
    fieldno = daWEEKDAY;
  } else if (boost::iequals(field, "decade") || boost::iequals(field, "dc")) {
    fieldno = daDECADE;
  } else if (boost::iequals(field, "century")) {
    fieldno = daCENTURY;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = daMILLENNIUM;
  } else {
    throw std::runtime_error("Unsupported field in DATEADD function: " + field);
  }
  return fieldno;
}

DatetruncField to_datediff_field(const std::string& field) {
  DatetruncField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = dtYEAR;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = dtQUARTER;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = dtMONTH;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = dtWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = dtWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = dtWEEK_SATURDAY;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = dtDAY;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = dtQUARTERDAY;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = dtHOUR;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = dtMINUTE;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = dtSECOND;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = dtMILLISECOND;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = dtMICROSECOND;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = dtNANOSECOND;
  } else if (boost::iequals(field, "decade") || boost::iequals(field, "dc")) {
    fieldno = dtDECADE;
  } else if (boost::iequals(field, "century")) {
    fieldno = dtCENTURY;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = dtMILLENNIUM;
  } else {
    throw std::runtime_error("Unsupported field in DATEDIFF function: " + field);
  }
  return fieldno;
}

std::shared_ptr<Analyzer::Constant> make_fp_constant(const int64_t val,
                                                     const SQLTypeInfo& ti) {
  Datum d;
  switch (ti.get_type()) {
    case kFLOAT:
      d.floatval = val;
      break;
    case kDOUBLE:
      d.doubleval = val;
      break;
    default:
      CHECK(false);
  }
  return makeExpr<Analyzer::Constant>(ti, false, d);
}
