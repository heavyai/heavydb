/*
 * Copyright 2019 OmniSci, Inc.
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

#include "DateTimeTranslator.h"

std::string ExtractExpr::from_extract_field(const ExtractField& fieldno) {
  switch (fieldno) {
    case kYEAR:
      return "year";
    case kQUARTER:
      return "quarter";
    case kMONTH:
      return "month";
    case kDAY:
      return "day";
    case kHOUR:
      return "hour";
    case kMINUTE:
      return "minute";
    case kSECOND:
      return "second";
    case kMILLISECOND:
      return "millisecond";
    case kMICROSECOND:
      return "microsecond";
    case kNANOSECOND:
      return "nanosecond";
    case kDOW:
      return "dow";
    case kISODOW:
      return "isodow";
    case kDOY:
      return "doy";
    case kEPOCH:
      return "epoch";
    case kQUARTERDAY:
      return "quarterday";
    case kWEEK:
      return "week";
    default:
      UNREACHABLE();
  }
  return "";
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::generateExtractExpr(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const std::string& field_name) {
  const auto field = to_extract_field(field_name);
  return ExtractExpr::generateExtractExpr(from_expr, field);
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::generateExtractExpr(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const ExtractField& field) {
  const auto expr_ti = from_expr->get_type_info();
  if (!expr_ti.is_time()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in EXTRACT function.");
  }
  if (expr_ti.get_type() == kTIME && field != kHOUR && field != kMINUTE &&
      field != kSECOND) {
    throw std::runtime_error("Cannot EXTRACT " + from_extract_field(field) +
                             " from TIME.");
  }
  const SQLTypeInfo ti(kBIGINT, 0, 0, expr_ti.get_notnull());
  auto constant = std::dynamic_pointer_cast<Analyzer::Constant>(from_expr);
  if (constant != nullptr) {
    Datum d;
    d.bigintval = field == kEPOCH
                      ? constant->get_constval().bigintval
                      : getExtractFromTimeConstantValue(
                            constant->get_constval().bigintval, field, expr_ti);
    constant->set_constval(d);
    constant->set_type_info(ti);
    return constant;
  }
  if (!expr_ti.is_high_precision_timestamp() && is_subsecond_extract_field(field)) {
    return makeExpr<Analyzer::Constant>(ti, false, Datum{0});
  }
  return makeExpr<Analyzer::ExtractExpr>(
      ti, from_expr->get_contains_agg(), field, from_expr->decompress());
}

std::string DateTruncExpr::from_datetrunc_field(const DatetruncField& fieldno) {
  switch (fieldno) {
    case dtYEAR:
      return "year";
    case dtQUARTER:
      return "quarter";
    case dtMONTH:
      return "month";
    case dtQUARTERDAY:
      return "quarterday";
    case dtDAY:
      return "day";
    case dtHOUR:
      return "hour";
    case dtMINUTE:
      return "minute";
    case dtSECOND:
      return "second";
    case dtMILLENNIUM:
      return "millennium";
    case dtCENTURY:
      return "century";
    case dtDECADE:
      return "decade";
    case dtMILLISECOND:
      return "millisecond";
    case dtMICROSECOND:
      return "microsecond";
    case dtNANOSECOND:
      return "nanosecond";
    case dtWEEK:
      return "week";
    case dtINVALID:
    default:
      UNREACHABLE();
  }
  return "";
}

std::shared_ptr<Analyzer::Expr> DateTruncExpr::generateDatetruncExpr(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const std::string& field_name) {
  const auto field = to_datetrunc_field(field_name);
  return DateTruncExpr::generateDatetruncExpr(from_expr, field);
}

std::shared_ptr<Analyzer::Expr> DateTruncExpr::generateDatetruncExpr(
    const std::shared_ptr<Analyzer::Expr> from_expr,
    const DatetruncField& field) {
  const auto& expr_ti = from_expr->get_type_info();
  if (!expr_ti.is_time()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in DATE_TRUNC function.");
  }
  if (from_expr->get_type_info().get_type() == kTIME && field != dtHOUR &&
      field != dtMINUTE && field != dtSECOND) {
    throw std::runtime_error("Cannot DATE_TRUNC " + from_datetrunc_field(field) +
                             " from TIME.");
  }
  SQLTypeInfo ti(kTIMESTAMP, expr_ti.get_dimension(), 0, expr_ti.get_notnull());
  auto constant = std::dynamic_pointer_cast<Analyzer::Constant>(from_expr);
  if (constant) {
    Datum d{0};
    d.bigintval =
        getDateTruncConstantValue(constant->get_constval().bigintval, field, expr_ti);
    constant->set_constval(d);
    constant->set_type_info(ti);
    return constant;
  }
  return makeExpr<Analyzer::DatetruncExpr>(
      ti, from_expr->get_contains_agg(), field, from_expr->decompress());
}
