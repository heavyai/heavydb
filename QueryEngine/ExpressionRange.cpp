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

#include "ExpressionRange.h"
#include "DateTimeTranslator.h"
#include "DateTimeUtils.h"
#include "DateTruncate.h"
#include "Descriptors/InputDescriptors.h"
#include "Execute.h"
#include "ExtractFromTime.h"
#include "GroupByAndAggregate.h"
#include "QueryPhysicalInputsCollector.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

#define DEF_OPERATOR(fname, op)                                                    \
  ExpressionRange fname(const ExpressionRange& other) const {                      \
    if (type_ == ExpressionRangeType::Invalid ||                                   \
        other.type_ == ExpressionRangeType::Invalid) {                             \
      return ExpressionRange::makeInvalidRange();                                  \
    }                                                                              \
    CHECK(type_ == other.type_);                                                   \
    switch (type_) {                                                               \
      case ExpressionRangeType::Integer:                                           \
        return binOp<int64_t>(other, [](const int64_t x, const int64_t y) {        \
          return int64_t(checked_int64_t(x) op y);                                 \
        });                                                                        \
      case ExpressionRangeType::Float:                                             \
        return binOp<float>(other, [](const float x, const float y) {              \
          std::feclearexcept(FE_OVERFLOW);                                         \
          std::feclearexcept(FE_UNDERFLOW);                                        \
          auto result = x op y;                                                    \
          if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) { \
            throw std::runtime_error("overflow / underflow");                      \
          }                                                                        \
          return result;                                                           \
        });                                                                        \
      case ExpressionRangeType::Double:                                            \
        return binOp<double>(other, [](const double x, const double y) {           \
          std::feclearexcept(FE_OVERFLOW);                                         \
          std::feclearexcept(FE_UNDERFLOW);                                        \
          auto result = x op y;                                                    \
          if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) { \
            throw std::runtime_error("overflow / underflow");                      \
          }                                                                        \
          return result;                                                           \
        });                                                                        \
      default:                                                                     \
        CHECK(false);                                                              \
    }                                                                              \
    CHECK(false);                                                                  \
    return ExpressionRange::makeInvalidRange();                                    \
  }

DEF_OPERATOR(ExpressionRange::operator+, +)
DEF_OPERATOR(ExpressionRange::operator-, -)
DEF_OPERATOR(ExpressionRange::operator*, *)

void apply_fp_qual(const Datum const_datum,
                   const SQLTypes const_type,
                   const SQLOps sql_op,
                   ExpressionRange& qual_range) {
  double const_val = get_value_from_datum<double>(const_datum, const_type);
  switch (sql_op) {
    case kGT:
    case kGE:
      qual_range.setFpMin(std::max(qual_range.getFpMin(), const_val));
      break;
    case kLT:
    case kLE:
      qual_range.setFpMax(std::min(qual_range.getFpMax(), const_val));
      break;
    case kEQ:
      qual_range.setFpMin(std::max(qual_range.getFpMin(), const_val));
      qual_range.setFpMax(std::min(qual_range.getFpMax(), const_val));
      break;
    default:  // there may be other operators, but don't do anything with them
      break;
  }
}

void apply_int_qual(const Datum const_datum,
                    const SQLTypes const_type,
                    const SQLOps sql_op,
                    ExpressionRange& qual_range) {
  int64_t const_val = get_value_from_datum<int64_t>(const_datum, const_type);
  switch (sql_op) {
    case kGT:
      qual_range.setIntMin(std::max(qual_range.getIntMin(), const_val + 1));
      break;
    case kGE:
      qual_range.setIntMin(std::max(qual_range.getIntMin(), const_val));
      break;
    case kLT:
      qual_range.setIntMax(std::min(qual_range.getIntMax(), const_val - 1));
      break;
    case kLE:
      qual_range.setIntMax(std::min(qual_range.getIntMax(), const_val));
      break;
    case kEQ:
      qual_range.setIntMin(std::max(qual_range.getIntMin(), const_val));
      qual_range.setIntMax(std::min(qual_range.getIntMax(), const_val));
      break;
    default:  // there may be other operators, but don't do anything with them
      break;
  }
}

void apply_hpt_qual(const Datum const_datum,
                    const SQLTypes const_type,
                    const int32_t const_dimen,
                    const int32_t col_dimen,
                    const SQLOps sql_op,
                    ExpressionRange& qual_range) {
  CHECK(const_dimen != col_dimen);
  Datum datum{0};
  if (const_dimen > col_dimen) {
    datum.bigintval =
        get_value_from_datum<int64_t>(const_datum, const_type) /
        DateTimeUtils::get_timestamp_precision_scale(const_dimen - col_dimen);
  } else {
    datum.bigintval =
        get_value_from_datum<int64_t>(const_datum, const_type) *
        DateTimeUtils::get_timestamp_precision_scale(col_dimen - const_dimen);
  }
  apply_int_qual(datum, const_type, sql_op, qual_range);
}

ExpressionRange apply_simple_quals(
    const Analyzer::ColumnVar* col_expr,
    const ExpressionRange& col_range,
    const boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  if (!simple_quals) {
    return col_range;
  }
  ExpressionRange qual_range(col_range);
  for (auto const& itr : simple_quals.get()) {
    auto qual_bin_oper = dynamic_cast<Analyzer::BinOper*>(itr.get());
    if (!qual_bin_oper) {
      continue;
    }
    const Analyzer::Expr* left_operand = qual_bin_oper->get_left_operand();
    auto qual_col = dynamic_cast<const Analyzer::ColumnVar*>(left_operand);
    if (!qual_col) {
      // Check for possibility that column is wrapped in a cast
      // Presumes that only simple casts (i.e. timestamp to timestamp or int to int) have
      // been passed through by BinOper::normalize_simple_predicate
      auto u_expr = dynamic_cast<const Analyzer::UOper*>(left_operand);
      if (!u_expr) {
        continue;
      }
      qual_col = dynamic_cast<const Analyzer::ColumnVar*>(u_expr->get_operand());
      if (!qual_col) {
        continue;
      }
    }
    if (qual_col->get_table_id() != col_expr->get_table_id() ||
        qual_col->get_column_id() != col_expr->get_column_id()) {
      continue;
    }
    const Analyzer::Expr* right_operand = qual_bin_oper->get_right_operand();
    auto qual_const = dynamic_cast<const Analyzer::Constant*>(right_operand);
    if (!qual_const) {
      continue;
    }
    if (qual_range.getType() == ExpressionRangeType::Float ||
        qual_range.getType() == ExpressionRangeType::Double) {
      apply_fp_qual(qual_const->get_constval(),
                    qual_const->get_type_info().get_type(),
                    qual_bin_oper->get_optype(),
                    qual_range);
    } else if ((qual_col->get_type_info().is_timestamp() ||
                qual_const->get_type_info().is_timestamp()) &&
               (qual_col->get_type_info().get_dimension() !=
                qual_const->get_type_info().get_dimension())) {
      apply_hpt_qual(qual_const->get_constval(),
                     qual_const->get_type_info().get_type(),
                     qual_const->get_type_info().get_dimension(),
                     qual_col->get_type_info().get_dimension(),
                     qual_bin_oper->get_optype(),
                     qual_range);
    } else {
      apply_int_qual(qual_const->get_constval(),
                     qual_const->get_type_info().get_type(),
                     qual_bin_oper->get_optype(),
                     qual_range);
    }
  }
  return qual_range;
}

ExpressionRange ExpressionRange::operator/(const ExpressionRange& other) const {
  if (type_ != ExpressionRangeType::Integer ||
      other.type_ != ExpressionRangeType::Integer) {
    return ExpressionRange::makeInvalidRange();
  }
  if (other.int_min_ * other.int_max_ <= 0) {
    // if the other interval contains 0, the rule is more complicated;
    // punt for now, we can revisit by splitting the other interval and
    // taking the convex hull of the resulting two intervals
    return ExpressionRange::makeInvalidRange();
  }
  auto div_range = binOp<int64_t>(other, [](const int64_t x, const int64_t y) {
    return int64_t(checked_int64_t(x) / y);
  });
  if (g_null_div_by_zero) {
    div_range.setHasNulls();
  }
  return div_range;
}

ExpressionRange ExpressionRange::operator||(const ExpressionRange& other) const {
  if (type_ != other.type_) {
    return ExpressionRange::makeInvalidRange();
  }
  ExpressionRange result;
  switch (type_) {
    case ExpressionRangeType::Invalid:
      return ExpressionRange::makeInvalidRange();
    case ExpressionRangeType::Integer: {
      result.type_ = ExpressionRangeType::Integer;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      result.int_min_ = std::min(int_min_, other.int_min_);
      result.int_max_ = std::max(int_max_, other.int_max_);
      result.bucket_ = std::min(bucket_, other.bucket_);
      break;
    }
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      result.type_ = type_;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      result.fp_min_ = std::min(fp_min_, other.fp_min_);
      result.fp_max_ = std::max(fp_max_, other.fp_max_);
      break;
    }
    default:
      CHECK(false);
  }
  return result;
}

bool ExpressionRange::operator==(const ExpressionRange& other) const {
  if (type_ != other.type_) {
    return false;
  }
  switch (type_) {
    case ExpressionRangeType::Invalid:
      return true;
    case ExpressionRangeType::Integer: {
      return has_nulls_ == other.has_nulls_ && int_min_ == other.int_min_ &&
             int_max_ == other.int_max_;
    }
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      return has_nulls_ == other.has_nulls_ && fp_min_ == other.fp_min_ &&
             fp_max_ == other.fp_max_;
    }
    default:
      CHECK(false);
  }
  return false;
}

bool ExpressionRange::typeSupportsRange(const SQLTypeInfo& ti) {
  if (ti.is_array()) {
    return typeSupportsRange(ti.get_elem_type());
  } else {
    return (ti.is_number() || ti.is_boolean() || ti.is_time() ||
            (ti.is_string() && ti.get_compression() == kENCODING_DICT));
  }
}

ExpressionRange getExpressionRange(
    const Analyzer::BinOper* expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor*,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals);

ExpressionRange getExpressionRange(const Analyzer::Constant* expr);

ExpressionRange getExpressionRange(
    const Analyzer::ColumnVar* col_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals);

ExpressionRange getExpressionRange(const Analyzer::StringOper* string_oper_expr,
                                   const Executor* executor);

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr);

ExpressionRange getExpressionRange(const Analyzer::CaseExpr* case_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(
    const Analyzer::UOper* u_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor*,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals);

ExpressionRange getExpressionRange(
    const Analyzer::ExtractExpr* extract_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor*,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals);

ExpressionRange getExpressionRange(
    const Analyzer::DatetruncExpr* datetrunc_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals);

ExpressionRange getExpressionRange(
    const Analyzer::WidthBucketExpr* width_bucket_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals);

ExpressionRange getExpressionRange(
    const Analyzer::Expr* expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  if (!ExpressionRange::typeSupportsRange(expr->get_type_info())) {
    return ExpressionRange::makeInvalidRange();
  } else if (auto bin_oper_expr = dynamic_cast<const Analyzer::BinOper*>(expr)) {
    return getExpressionRange(bin_oper_expr, query_infos, executor, simple_quals);
  } else if (auto constant_expr = dynamic_cast<const Analyzer::Constant*>(expr)) {
    return getExpressionRange(constant_expr);
  } else if (auto column_var_expr = dynamic_cast<const Analyzer::ColumnVar*>(expr)) {
    return getExpressionRange(column_var_expr, query_infos, executor, simple_quals);
  } else if (auto string_oper_expr = dynamic_cast<const Analyzer::StringOper*>(expr)) {
    return getExpressionRange(string_oper_expr, executor);
  } else if (auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr)) {
    return getExpressionRange(like_expr);
  } else if (auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr)) {
    return getExpressionRange(case_expr, query_infos, executor);
  } else if (auto u_expr = dynamic_cast<const Analyzer::UOper*>(expr)) {
    return getExpressionRange(u_expr, query_infos, executor, simple_quals);
  } else if (auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr)) {
    return getExpressionRange(extract_expr, query_infos, executor, simple_quals);
  } else if (auto datetrunc_expr = dynamic_cast<const Analyzer::DatetruncExpr*>(expr)) {
    return getExpressionRange(datetrunc_expr, query_infos, executor, simple_quals);
  } else if (auto width_expr = dynamic_cast<const Analyzer::WidthBucketExpr*>(expr)) {
    return getExpressionRange(width_expr, query_infos, executor, simple_quals);
  }
  return ExpressionRange::makeInvalidRange();
}

namespace {

int64_t scale_up_interval_endpoint(const int64_t endpoint, const SQLTypeInfo& ti) {
  return endpoint * static_cast<int64_t>(exp_to_scale(ti.get_scale()));
}

}  // namespace

ExpressionRange getExpressionRange(
    const Analyzer::BinOper* expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  const auto& lhs =
      getExpressionRange(expr->get_left_operand(), query_infos, executor, simple_quals);
  const auto& rhs =
      getExpressionRange(expr->get_right_operand(), query_infos, executor, simple_quals);
  switch (expr->get_optype()) {
    case kPLUS:
      return lhs + rhs;
    case kMINUS:
      return lhs - rhs;
    case kMULTIPLY:
      return lhs * rhs;
    case kDIVIDE: {
      const auto& lhs_type = expr->get_left_operand()->get_type_info();
      if (lhs_type.is_decimal() && lhs.getType() != ExpressionRangeType::Invalid) {
        CHECK(lhs.getType() == ExpressionRangeType::Integer);
        const auto adjusted_lhs = ExpressionRange::makeIntRange(
            scale_up_interval_endpoint(lhs.getIntMin(), lhs_type),
            scale_up_interval_endpoint(lhs.getIntMax(), lhs_type),
            0,
            lhs.hasNulls());
        return adjusted_lhs / rhs;
      }
      return lhs / rhs;
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(const Analyzer::Constant* constant_expr) {
  if (constant_expr->get_is_null()) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto constant_type = constant_expr->get_type_info().get_type();
  const auto datum = constant_expr->get_constval();
  switch (constant_type) {
    case kTINYINT: {
      const int64_t v = datum.tinyintval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kSMALLINT: {
      const int64_t v = datum.smallintval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kINT: {
      const int64_t v = datum.intval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL: {
      const int64_t v = datum.bigintval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      const int64_t v = datum.bigintval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kFLOAT: {
      return ExpressionRange::makeFloatRange(datum.floatval, datum.floatval, false);
    }
    case kDOUBLE: {
      return ExpressionRange::makeDoubleRange(datum.doubleval, datum.doubleval, false);
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

#define FIND_STAT_FRAG(stat_name)                                                    \
  const auto stat_name##_frag_index = std::stat_name##_element(                      \
      nonempty_fragment_indices.begin(),                                             \
      nonempty_fragment_indices.end(),                                               \
      [&fragments, &has_nulls, col_id, col_ti](const size_t lhs_idx,                 \
                                               const size_t rhs_idx) {               \
        const auto& lhs = fragments[lhs_idx];                                        \
        const auto& rhs = fragments[rhs_idx];                                        \
        auto lhs_meta_it = lhs.getChunkMetadataMap().find(col_id);                   \
        if (lhs_meta_it == lhs.getChunkMetadataMap().end()) {                        \
          return false;                                                              \
        }                                                                            \
        auto rhs_meta_it = rhs.getChunkMetadataMap().find(col_id);                   \
        CHECK(rhs_meta_it != rhs.getChunkMetadataMap().end());                       \
        if (lhs_meta_it->second->chunkStats.has_nulls ||                             \
            rhs_meta_it->second->chunkStats.has_nulls) {                             \
          has_nulls = true;                                                          \
        }                                                                            \
        if (col_ti.is_fp()) {                                                        \
          return extract_##stat_name##_stat_fp_type(lhs_meta_it->second->chunkStats, \
                                                    col_ti) <                        \
                 extract_##stat_name##_stat_fp_type(rhs_meta_it->second->chunkStats, \
                                                    col_ti);                         \
        }                                                                            \
        return extract_##stat_name##_stat_int_type(lhs_meta_it->second->chunkStats,  \
                                                   col_ti) <                         \
               extract_##stat_name##_stat_int_type(rhs_meta_it->second->chunkStats,  \
                                                   col_ti);                          \
      });                                                                            \
  if (stat_name##_frag_index == nonempty_fragment_indices.end()) {                   \
    return ExpressionRange::makeInvalidRange();                                      \
  }

namespace {

int64_t get_conservative_datetrunc_bucket(const DatetruncField datetrunc_field) {
  const int64_t day_seconds{24 * 3600};
  const int64_t year_days{365};
  switch (datetrunc_field) {
    case dtYEAR:
      return year_days * day_seconds;
    case dtQUARTER:
      return 90 * day_seconds;  // 90 is least number of days in any quater
    case dtMONTH:
      return 28 * day_seconds;
    case dtDAY:
      return day_seconds;
    case dtHOUR:
      return 3600;
    case dtMINUTE:
      return 60;
    case dtMILLENNIUM:
      return 1000 * year_days * day_seconds;
    case dtCENTURY:
      return 100 * year_days * day_seconds;
    case dtDECADE:
      return 10 * year_days * day_seconds;
    case dtWEEK:
    case dtWEEK_SUNDAY:
    case dtWEEK_SATURDAY:
      return 7 * day_seconds;
    case dtQUARTERDAY:
      return 4 * 60 * 50;
    default:
      return 0;
  }
}

}  // namespace

ExpressionRange getLeafColumnRange(const Analyzer::ColumnVar* col_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor,
                                   const bool is_outer_join_proj) {
  bool has_nulls = is_outer_join_proj;
  int col_id = col_expr->get_column_id();
  const auto& col_phys_ti = col_expr->get_type_info().is_array()
                                ? col_expr->get_type_info().get_elem_type()
                                : col_expr->get_type_info();
  const auto col_ti = get_logical_type_info(col_phys_ti);
  switch (col_ti.get_type()) {
    case kTEXT:
    case kCHAR:
    case kVARCHAR:
      CHECK_EQ(kENCODING_DICT, col_ti.get_compression());
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kDECIMAL:
    case kNUMERIC:
    case kDATE:
    case kTIMESTAMP:
    case kTIME:
    case kFLOAT:
    case kDOUBLE: {
      std::optional<size_t> ti_idx;
      for (size_t i = 0; i < query_infos.size(); ++i) {
        if (col_expr->get_table_id() == query_infos[i].table_id) {
          ti_idx = i;
          break;
        }
      }
      CHECK(ti_idx);
      const auto& query_info = query_infos[*ti_idx].info;
      const auto& fragments = query_info.fragments;
      const auto cd = executor->getColumnDescriptor(col_expr);
      if (cd && cd->isVirtualCol) {
        CHECK(cd->columnName == "rowid");
        CHECK_EQ(kBIGINT, col_ti.get_type());
        const int64_t num_tuples = query_info.getNumTuples();
        return ExpressionRange::makeIntRange(
            0, std::max(num_tuples - 1, int64_t(0)), 0, has_nulls);
      }
      if (query_info.getNumTuples() == 0) {
        // The column doesn't contain any values, synthesize an empty range.
        if (col_ti.is_fp()) {
          return col_ti.get_type() == kFLOAT
                     ? ExpressionRange::makeFloatRange(0, -1, false)
                     : ExpressionRange::makeDoubleRange(0, -1, false);
        }
        return ExpressionRange::makeIntRange(0, -1, 0, false);
      }
      std::vector<size_t> nonempty_fragment_indices;
      for (size_t i = 0; i < fragments.size(); ++i) {
        const auto& fragment = fragments[i];
        if (!fragment.isEmptyPhysicalFragment()) {
          nonempty_fragment_indices.push_back(i);
        }
      }
      FIND_STAT_FRAG(min);
      FIND_STAT_FRAG(max);
      const auto& min_frag = fragments[*min_frag_index];
      const auto min_it = min_frag.getChunkMetadataMap().find(col_id);
      if (min_it == min_frag.getChunkMetadataMap().end()) {
        return ExpressionRange::makeInvalidRange();
      }
      const auto& max_frag = fragments[*max_frag_index];
      const auto max_it = max_frag.getChunkMetadataMap().find(col_id);
      CHECK(max_it != max_frag.getChunkMetadataMap().end());
      for (const auto& fragment : fragments) {
        const auto it = fragment.getChunkMetadataMap().find(col_id);
        if (it != fragment.getChunkMetadataMap().end()) {
          if (it->second->chunkStats.has_nulls) {
            has_nulls = true;
            break;
          }
        }
      }

      // Detect FSI placeholder metadata.  If we let this through it will be treated as an
      // empty range, when it really implies an unknown range.
      if (min_it->second->isPlaceholder() || max_it->second->isPlaceholder()) {
        return ExpressionRange::makeInvalidRange();
      }

      if (col_ti.is_fp()) {
        const auto min_val = extract_min_stat_fp_type(min_it->second->chunkStats, col_ti);
        const auto max_val = extract_max_stat_fp_type(max_it->second->chunkStats, col_ti);
        return col_ti.get_type() == kFLOAT
                   ? ExpressionRange::makeFloatRange(min_val, max_val, has_nulls)
                   : ExpressionRange::makeDoubleRange(min_val, max_val, has_nulls);
      }
      const auto min_val = extract_min_stat_int_type(min_it->second->chunkStats, col_ti);
      const auto max_val = extract_max_stat_int_type(max_it->second->chunkStats, col_ti);
      if (max_val < min_val) {
        // The column doesn't contain any non-null values, synthesize an empty range.
        CHECK_GT(min_val, 0);
        return ExpressionRange::makeIntRange(0, -1, 0, has_nulls);
      }
      const int64_t bucket =
          col_ti.get_type() == kDATE ? get_conservative_datetrunc_bucket(dtDAY) : 0;
      return ExpressionRange::makeIntRange(min_val, max_val, bucket, has_nulls);
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

#undef FIND_STAT_FRAG

ExpressionRange getExpressionRange(
    const Analyzer::ColumnVar* col_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  const int rte_idx = col_expr->get_rte_idx();
  CHECK_GE(rte_idx, 0);
  CHECK_LT(static_cast<size_t>(rte_idx), query_infos.size());
  bool is_outer_join_proj = rte_idx > 0 && executor->containsLeftDeepOuterJoin();
  if (col_expr->get_table_id() > 0) {
    auto col_range = executor->getColRange(
        PhysicalInput{col_expr->get_column_id(), col_expr->get_table_id()});
    if (is_outer_join_proj) {
      col_range.setHasNulls();
    }
    return apply_simple_quals(col_expr, col_range, simple_quals);
  }
  return getLeafColumnRange(col_expr, query_infos, executor, is_outer_join_proj);
}

ExpressionRange getExpressionRange(const Analyzer::StringOper* string_oper_expr,
                                   const Executor* executor) {
  auto chained_string_op_exprs = string_oper_expr->getChainedStringOpExprs();
  if (chained_string_op_exprs.empty()) {
    throw std::runtime_error(
        "StringOper getExpressionRange() Expected folded string operator but found "
        "operator unfolded.");
  }
  // Consider encapsulating below in an Analyzer::StringOper method to dedup
  std::vector<StringOps_Namespace::StringOpInfo> string_op_infos;
  for (const auto& chained_string_op_expr : chained_string_op_exprs) {
    auto chained_string_op =
        dynamic_cast<const Analyzer::StringOper*>(chained_string_op_expr.get());
    CHECK(chained_string_op);
    StringOps_Namespace::StringOpInfo string_op_info(chained_string_op->get_kind(),
                                                     chained_string_op->get_type_info(),
                                                     chained_string_op->getLiteralArgs());
    string_op_infos.emplace_back(string_op_info);
  }

  const auto expr_ti = string_oper_expr->get_type_info();
  if (expr_ti.is_string() && !string_oper_expr->requiresPerRowTranslation()) {
    // We can only statically know the range if dictionary translation occured (which
    // means the output type is a dictionary-encoded text col and is not transient, as
    // transient dictionaries are used as targets for on-the-fly translation
    CHECK(expr_ti.is_dict_encoded_string());
    const auto dict_id = expr_ti.get_comp_param();
    CHECK_NE(dict_id, TRANSIENT_DICT_ID);
    const auto translation_map = executor->getStringProxyTranslationMap(
        dict_id,
        dict_id,
        RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
        string_op_infos,
        executor->getRowSetMemoryOwner(),
        true);

    // Todo(todd): Track null presence in StringDictionaryProxy::IdMap
    return ExpressionRange::makeIntRange(translation_map->rangeStart(),
                                         translation_map->rangeEnd() - 1,
                                         0 /* bucket */,
                                         true /* assume has nulls */);

  } else {
    // Todo(todd): Get min/max range stats
    // Below works fine, we just can't take advantage of low-cardinality
    // group by optimizations
    return ExpressionRange::makeInvalidRange();
  }
}

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr) {
  const auto& ti = like_expr->get_type_info();
  CHECK(ti.is_boolean());
  const auto& arg_ti = like_expr->get_arg()->get_type_info();
  return ExpressionRange::makeIntRange(
      arg_ti.get_notnull() ? 0 : inline_int_null_val(ti), 1, 0, false);
}

ExpressionRange getExpressionRange(const Analyzer::CaseExpr* case_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  auto expr_range = ExpressionRange::makeInvalidRange();
  bool has_nulls = false;
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().get_type(), kBOOLEAN);
    const auto crt_range =
        getExpressionRange(expr_pair.second.get(), query_infos, executor);
    if (crt_range.getType() == ExpressionRangeType::Null) {
      has_nulls = true;
      continue;
    }
    if (crt_range.getType() == ExpressionRangeType::Invalid) {
      return ExpressionRange::makeInvalidRange();
    }
    expr_range = (expr_range.getType() != ExpressionRangeType::Invalid)
                     ? expr_range || crt_range
                     : crt_range;
  }
  if (has_nulls && !(expr_range.getType() == ExpressionRangeType::Invalid)) {
    expr_range.setHasNulls();
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  const auto else_null_expr = dynamic_cast<const Analyzer::Constant*>(else_expr);
  if (else_null_expr && else_null_expr->get_is_null()) {
    expr_range.setHasNulls();
    return expr_range;
  }
  return expr_range || getExpressionRange(else_expr, query_infos, executor);
}

namespace {

ExpressionRange fpRangeFromDecimal(const ExpressionRange& arg_range,
                                   const int64_t scale,
                                   const SQLTypeInfo& target_ti) {
  CHECK(target_ti.is_fp());
  if (target_ti.get_type() == kFLOAT) {
    return ExpressionRange::makeFloatRange(
        static_cast<float>(arg_range.getIntMin()) / scale,
        static_cast<float>(arg_range.getIntMax()) / scale,
        arg_range.hasNulls());
  }
  return ExpressionRange::makeDoubleRange(
      static_cast<double>(arg_range.getIntMin()) / scale,
      static_cast<double>(arg_range.getIntMax()) / scale,
      arg_range.hasNulls());
}

ExpressionRange getDateTimePrecisionCastRange(const ExpressionRange& arg_range,
                                              const SQLTypeInfo& oper_ti,
                                              const SQLTypeInfo& target_ti) {
  if (oper_ti.is_timestamp() && target_ti.is_date()) {
    const auto field = dtDAY;
    const int64_t scale =
        oper_ti.is_high_precision_timestamp()
            ? DateTimeUtils::get_timestamp_precision_scale(oper_ti.get_dimension())
            : 1;
    const int64_t min_ts = oper_ti.is_high_precision_timestamp()
                               ? DateTruncate(field, arg_range.getIntMin() / scale)
                               : DateTruncate(field, arg_range.getIntMin());
    const int64_t max_ts = oper_ti.is_high_precision_timestamp()
                               ? DateTruncate(field, arg_range.getIntMax() / scale)
                               : DateTruncate(field, arg_range.getIntMax());
    const int64_t bucket = get_conservative_datetrunc_bucket(field);

    return ExpressionRange::makeIntRange(min_ts, max_ts, bucket, arg_range.hasNulls());
  }

  if (oper_ti.is_timestamp() && target_ti.is_any(kTIME)) {
    // The min and max TS wouldn't make sense here, so return a range covering the whole
    // day
    return ExpressionRange::makeIntRange(0, kSecsPerDay, 0, arg_range.hasNulls());
  }

  const int32_t ti_dimen = target_ti.get_dimension();
  const int32_t oper_dimen = oper_ti.get_dimension();
  CHECK(oper_dimen != ti_dimen);
  const int64_t min_ts =
      ti_dimen > oper_dimen
          ? DateTimeUtils::get_datetime_scaled_epoch(DateTimeUtils::ScalingType::ScaleUp,
                                                     arg_range.getIntMin(),
                                                     abs(oper_dimen - ti_dimen))
          : DateTimeUtils::get_datetime_scaled_epoch(
                DateTimeUtils::ScalingType::ScaleDown,
                arg_range.getIntMin(),
                abs(oper_dimen - ti_dimen));
  const int64_t max_ts =
      ti_dimen > oper_dimen
          ? DateTimeUtils::get_datetime_scaled_epoch(DateTimeUtils::ScalingType::ScaleUp,
                                                     arg_range.getIntMax(),
                                                     abs(oper_dimen - ti_dimen))
          : DateTimeUtils::get_datetime_scaled_epoch(
                DateTimeUtils::ScalingType::ScaleDown,
                arg_range.getIntMax(),
                abs(oper_dimen - ti_dimen));

  return ExpressionRange::makeIntRange(min_ts, max_ts, 0, arg_range.hasNulls());
}

}  // namespace

ExpressionRange getExpressionRange(
    const Analyzer::UOper* u_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  if (u_expr->get_optype() == kUNNEST) {
    return getExpressionRange(u_expr->get_operand(), query_infos, executor, simple_quals);
  }
  if (u_expr->get_optype() != kCAST) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto& ti = u_expr->get_type_info();
  if (ti.is_string() && ti.get_compression() == kENCODING_DICT) {
    const auto sdp = executor->getStringDictionaryProxy(
        ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
    CHECK(sdp);
    const auto colvar_operand =
        dynamic_cast<const Analyzer::ColumnVar*>(u_expr->get_operand());
    if (colvar_operand) {
      const auto& colvar_ti = colvar_operand->get_type_info();
      if (!(colvar_ti.is_none_encoded_string() &&
            ti.get_comp_param() == TRANSIENT_DICT_ID)) {
        VLOG(1)
            << "Unable to determine expression range for dictionary encoded expression "
            << u_expr->get_operand()->toString() << ", proceeding with invalid range.";
        return ExpressionRange::makeInvalidRange();
      }
      CHECK_EQ(ti.get_comp_param(), TRANSIENT_DICT_ID);
      CHECK_EQ(sdp->storageEntryCount(), 0UL);
      const int64_t transient_entries = static_cast<int64_t>(sdp->transientEntryCount());
      int64_t const tuples_upper_bound = static_cast<int64_t>(
          std::accumulate(query_infos.cbegin(),
                          query_infos.cend(),
                          size_t(0),
                          [](auto max, auto const& query_info) {
                            return std::max(max, query_info.info.getNumTuples());
                          }));
      const int64_t conservative_range_min = -1L - transient_entries - tuples_upper_bound;
      return ExpressionRange::makeIntRange(conservative_range_min, -2L, 0, true);
    }
    const auto const_operand =
        dynamic_cast<const Analyzer::Constant*>(u_expr->get_operand());
    if (!const_operand) {
      // casted subquery result. return invalid for now, but we could attempt to pull the
      // range from the subquery result in the future
      CHECK(u_expr->get_operand());
      VLOG(1) << "Unable to determine expression range for dictionary encoded expression "
              << u_expr->get_operand()->toString() << ", proceeding with invalid range.";
      return ExpressionRange::makeInvalidRange();
    }

    if (const_operand->get_is_null()) {
      return ExpressionRange::makeNullRange();
    }
    CHECK(const_operand->get_constval().stringval);
    const int64_t v = sdp->getIdOfString(*const_operand->get_constval().stringval);
    return ExpressionRange::makeIntRange(v, v, 0, false);
  }
  const auto arg_range =
      getExpressionRange(u_expr->get_operand(), query_infos, executor, simple_quals);
  const auto& arg_ti = u_expr->get_operand()->get_type_info();
  // Timestamp to Date OR Date/Timestamp casts with different precision
  if ((ti.is_timestamp() && (arg_ti.get_dimension() != ti.get_dimension())) ||
      ((arg_ti.is_timestamp() && ti.is_any(kDATE, kTIME)))) {
    return getDateTimePrecisionCastRange(arg_range, arg_ti, ti);
  }
  switch (arg_range.getType()) {
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      if (ti.is_fp()) {
        return ti.get_type() == kDOUBLE
                   ? ExpressionRange::makeDoubleRange(
                         arg_range.getFpMin(), arg_range.getFpMax(), arg_range.hasNulls())
                   : ExpressionRange::makeFloatRange(arg_range.getFpMin(),
                                                     arg_range.getFpMax(),
                                                     arg_range.hasNulls());
      }
      if (ti.is_integer()) {
        return ExpressionRange::makeIntRange(std::floor(arg_range.getFpMin()),
                                             std::ceil(arg_range.getFpMax()),
                                             0,
                                             arg_range.hasNulls());
      }
      break;
    }
    case ExpressionRangeType::Integer: {
      if (ti.is_decimal()) {
        CHECK_EQ(int64_t(0), arg_range.getBucket());
        const int64_t scale = exp_to_scale(ti.get_scale() - arg_ti.get_scale());
        return ExpressionRange::makeIntRange(arg_range.getIntMin() * scale,
                                             arg_range.getIntMax() * scale,
                                             0,
                                             arg_range.hasNulls());
      }
      if (arg_ti.is_decimal()) {
        CHECK_EQ(int64_t(0), arg_range.getBucket());
        const int64_t scale = exp_to_scale(arg_ti.get_scale());
        const int64_t scale_half = scale / 2;
        if (ti.is_fp()) {
          return fpRangeFromDecimal(arg_range, scale, ti);
        }
        return ExpressionRange::makeIntRange((arg_range.getIntMin() - scale_half) / scale,
                                             (arg_range.getIntMax() + scale_half) / scale,
                                             0,
                                             arg_range.hasNulls());
      }
      if (ti.is_integer() || ti.is_time()) {
        return arg_range;
      }
      if (ti.get_type() == kFLOAT) {
        return ExpressionRange::makeFloatRange(
            arg_range.getIntMin(), arg_range.getIntMax(), arg_range.hasNulls());
      }
      if (ti.get_type() == kDOUBLE) {
        return ExpressionRange::makeDoubleRange(
            arg_range.getIntMin(), arg_range.getIntMax(), arg_range.hasNulls());
      }
      break;
    }
    case ExpressionRangeType::Invalid:
      break;
    default:
      CHECK(false);
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(
    const Analyzer::ExtractExpr* extract_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  const int32_t extract_field{extract_expr->get_field()};
  const auto arg_range = getExpressionRange(
      extract_expr->get_from_expr(), query_infos, executor, simple_quals);
  const bool has_nulls =
      arg_range.getType() == ExpressionRangeType::Invalid || arg_range.hasNulls();
  const auto& extract_expr_ti = extract_expr->get_from_expr()->get_type_info();
  switch (extract_field) {
    case kYEAR: {
      if (arg_range.getType() == ExpressionRangeType::Invalid) {
        return ExpressionRange::makeInvalidRange();
      }
      CHECK(arg_range.getType() == ExpressionRangeType::Integer);
      const int64_t year_range_min =
          extract_expr_ti.is_high_precision_timestamp()
              ? ExtractFromTime(
                    kYEAR,
                    arg_range.getIntMin() /
                        get_timestamp_precision_scale(extract_expr_ti.get_dimension()))
              : ExtractFromTime(kYEAR, arg_range.getIntMin());
      const int64_t year_range_max =
          extract_expr_ti.is_high_precision_timestamp()
              ? ExtractFromTime(
                    kYEAR,
                    arg_range.getIntMax() /
                        get_timestamp_precision_scale(extract_expr_ti.get_dimension()))
              : ExtractFromTime(kYEAR, arg_range.getIntMax());
      return ExpressionRange::makeIntRange(
          year_range_min, year_range_max, 0, arg_range.hasNulls());
    }
    case kEPOCH:
    case kDATEEPOCH:
      return arg_range;
    case kQUARTERDAY:
    case kQUARTER:
      return ExpressionRange::makeIntRange(1, 4, 0, has_nulls);
    case kMONTH:
      return ExpressionRange::makeIntRange(1, 12, 0, has_nulls);
    case kDAY:
      return ExpressionRange::makeIntRange(1, 31, 0, has_nulls);
    case kHOUR:
      return ExpressionRange::makeIntRange(0, 23, 0, has_nulls);
    case kMINUTE:
      return ExpressionRange::makeIntRange(0, 59, 0, has_nulls);
    case kSECOND:
      return ExpressionRange::makeIntRange(0, 60, 0, has_nulls);
    case kMILLISECOND:
      return ExpressionRange::makeIntRange(0, 999, 0, has_nulls);
    case kMICROSECOND:
      return ExpressionRange::makeIntRange(0, 999999, 0, has_nulls);
    case kNANOSECOND:
      return ExpressionRange::makeIntRange(0, 999999999, 0, has_nulls);
    case kDOW:
      return ExpressionRange::makeIntRange(0, 6, 0, has_nulls);
    case kISODOW:
      return ExpressionRange::makeIntRange(1, 7, 0, has_nulls);
    case kDOY:
      return ExpressionRange::makeIntRange(1, 366, 0, has_nulls);
    case kWEEK:
    case kWEEK_SUNDAY:
    case kWEEK_SATURDAY:
      return ExpressionRange::makeIntRange(1, 53, 0, has_nulls);
    default:
      CHECK(false);
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(
    const Analyzer::DatetruncExpr* datetrunc_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  const auto arg_range = getExpressionRange(
      datetrunc_expr->get_from_expr(), query_infos, executor, simple_quals);
  if (arg_range.getType() == ExpressionRangeType::Invalid) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto& datetrunc_expr_ti = datetrunc_expr->get_from_expr()->get_type_info();
  const int64_t min_ts = DateTimeTranslator::getDateTruncConstantValue(
      arg_range.getIntMin(), datetrunc_expr->get_field(), datetrunc_expr_ti);
  const int64_t max_ts = DateTimeTranslator::getDateTruncConstantValue(
      arg_range.getIntMax(), datetrunc_expr->get_field(), datetrunc_expr_ti);
  const int64_t bucket =
      datetrunc_expr_ti.is_high_precision_timestamp()
          ? get_conservative_datetrunc_bucket(datetrunc_expr->get_field()) *
                DateTimeUtils::get_timestamp_precision_scale(
                    datetrunc_expr_ti.get_dimension())
          : get_conservative_datetrunc_bucket(datetrunc_expr->get_field());

  return ExpressionRange::makeIntRange(min_ts, max_ts, bucket, arg_range.hasNulls());
}

ExpressionRange getExpressionRange(
    const Analyzer::WidthBucketExpr* width_bucket_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> simple_quals) {
  auto target_value_expr = width_bucket_expr->get_target_value();
  auto target_value_range = getExpressionRange(target_value_expr, query_infos, executor);
  auto target_ti = target_value_expr->get_type_info();
  if (width_bucket_expr->is_constant_expr() &&
      target_value_range.getType() != ExpressionRangeType::Invalid) {
    auto const_target_value = dynamic_cast<const Analyzer::Constant*>(target_value_expr);
    if (const_target_value) {
      if (const_target_value->get_is_null()) {
        // null constant, return default width_bucket range
        return ExpressionRange::makeIntRange(
            0, width_bucket_expr->get_partition_count_val(), 0, true);
      } else {
        CHECK(target_value_range.getFpMax() == target_value_range.getFpMin());
        auto target_value_bucket =
            width_bucket_expr->compute_bucket(target_value_range.getFpMax(), target_ti);
        return ExpressionRange::makeIntRange(
            target_value_bucket, target_value_bucket, 0, target_value_range.hasNulls());
      }
    }
    // compute possible bucket range based on lower and upper bound constants
    // to elucidate a target bucket range
    const auto target_value_range_with_qual =
        getExpressionRange(target_value_expr, query_infos, executor, simple_quals);
    auto compute_bucket_range = [&width_bucket_expr](const ExpressionRange& target_range,
                                                     SQLTypeInfo ti) {
      // we casted bucket bound exprs to double
      auto lower_bound_bucket =
          width_bucket_expr->compute_bucket<double>(target_range.getFpMin(), ti);
      auto upper_bound_bucket =
          width_bucket_expr->compute_bucket<double>(target_range.getFpMax(), ti);
      return ExpressionRange::makeIntRange(
          lower_bound_bucket, upper_bound_bucket, 0, target_range.hasNulls());
    };
    auto res_range = compute_bucket_range(target_value_range_with_qual, target_ti);
    // check target_value expression's col range to be not nullable iff it has its filter
    // expression i.e., in simple_quals
    // todo (yoonmin) : need to search simple_quals to cover more cases?
    if (target_value_range.getFpMin() < target_value_range_with_qual.getFpMin() ||
        target_value_range.getFpMax() > target_value_range_with_qual.getFpMax()) {
      res_range.setNulls(false);
    }
    return res_range;
  } else {
    // we cannot determine a possibility of skipping oob check safely
    const bool has_nulls = target_value_range.getType() == ExpressionRangeType::Invalid ||
                           target_value_range.hasNulls();
    auto partition_expr_range = getExpressionRange(
        width_bucket_expr->get_partition_count(), query_infos, executor, simple_quals);
    auto res = ExpressionRange::makeIntRange(0, INT32_MAX, 0, has_nulls);
    switch (partition_expr_range.getType()) {
      case ExpressionRangeType::Integer: {
        res.setIntMax(partition_expr_range.getIntMax() + 1);
        break;
      }
      case ExpressionRangeType::Float:
      case ExpressionRangeType::Double: {
        res.setIntMax(static_cast<int64_t>(partition_expr_range.getFpMax()) + 1);
        break;
      }
      default:
        break;
    }
    return res;
  }
}
