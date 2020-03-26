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

#pragma once

#include "Analyzer/Analyzer.h"
#include "QueryEngine/ExtractFromTime.h"

#include "DateTimeUtils.h"

#include <memory>
#include <string>

using namespace DateTimeUtils;

class DateTimeTranslator {
 public:
  static inline int64_t getExtractFromTimeConstantValue(const int64_t& timeval,
                                                        const ExtractField& field,
                                                        const SQLTypeInfo& ti) {
    if (ti.is_high_precision_timestamp()) {
      if (is_subsecond_extract_field(field)) {
        const auto result =
            get_extract_high_precision_adjusted_scale(field, ti.get_dimension());
        return result.second
                   ? ExtractFromTime(field,
                                     result.first == kDIVIDE ? timeval / result.second
                                                             : timeval * result.second)
                   : ExtractFromTime(field, timeval);
      } else {
        return ExtractFromTime(
            field, timeval / get_timestamp_precision_scale(ti.get_dimension()));
      }
    } else if (is_subsecond_extract_field(field)) {
      return ExtractFromTime(field,
                             timeval * get_extract_timestamp_precision_scale(field));
    }
    return ExtractFromTime(field, timeval);
  }

  static inline int64_t getDateTruncConstantValue(const int64_t& timeval,
                                                  const DatetruncField& field,
                                                  const SQLTypeInfo& ti) {
    if (ti.is_high_precision_timestamp()) {
      if (is_subsecond_datetrunc_field(field)) {
        const auto result = get_datetrunc_high_precision_scale(field, ti.get_dimension());
        return result != -1 ? (DateTruncate(field, timeval) / result) * result
                            : DateTruncate(field, timeval);
      } else {
        const int64_t scale = get_timestamp_precision_scale(ti.get_dimension());
        return DateTruncate(field, timeval / scale) * scale;
      }
    }
    return DateTruncate(field, timeval);
  }

 protected:
  static inline std::shared_ptr<Analyzer::Constant> getNumericConstant(
      const int64_t scale) {
    Datum d{0};
    d.bigintval = scale;
    return makeExpr<Analyzer::Constant>(SQLTypeInfo(kBIGINT, false), false, d);
  }
};

class ExtractExpr : protected DateTimeTranslator {
 public:
  ExtractExpr(const std::shared_ptr<Analyzer::Expr> expr, const ExtractField& field)
      : from_expr_(expr), field_(field) {}
  ExtractExpr(const std::shared_ptr<Analyzer::Expr> expr, const std::string& field)
      : from_expr_(expr), field_(to_extract_field(field)) {}

  static std::shared_ptr<Analyzer::Expr> generate(const std::shared_ptr<Analyzer::Expr>,
                                                  const std::string&);
  static std::shared_ptr<Analyzer::Expr> generate(const std::shared_ptr<Analyzer::Expr>,
                                                  const ExtractField&);

  const std::shared_ptr<Analyzer::Expr> generate() const {
    return generate(from_expr_, field_);
  }

 private:
  static ExtractField to_extract_field(const std::string& field);

  std::shared_ptr<Analyzer::Expr> from_expr_;
  ExtractField field_;
};

class DateTruncExpr : protected DateTimeTranslator {
 public:
  DateTruncExpr(const std::shared_ptr<Analyzer::Expr> expr, const DatetruncField& field)
      : from_expr_(expr), field_(field) {}
  DateTruncExpr(const std::shared_ptr<Analyzer::Expr> expr, const std::string& field)
      : from_expr_(expr), field_(to_datetrunc_field(field)) {}

  static std::shared_ptr<Analyzer::Expr> generate(const std::shared_ptr<Analyzer::Expr>,
                                                  const std::string&);
  static std::shared_ptr<Analyzer::Expr> generate(const std::shared_ptr<Analyzer::Expr>,
                                                  const DatetruncField&);

  const std::shared_ptr<Analyzer::Expr> generate() const {
    return generate(from_expr_, field_);
  }

 private:
  static DatetruncField to_datetrunc_field(const std::string& field);

  std::shared_ptr<Analyzer::Expr> from_expr_;
  DatetruncField field_;
};
