#include "ExpressionRange.h"
#include "GroupByAndAggregate.h"

#include <cfenv>


#define DEF_OPERATOR(fname, op)                                                                                 \
ExpressionRange fname(const ExpressionRange& other) const {                                                     \
  return (type == ExpressionRangeType::Integer && other.type == ExpressionRangeType::Integer)                   \
    ? binOp<int64_t>(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) op y); })  \
    : binOp<double>(other, [](const double x, const double y) {                                                 \
      std::feclearexcept(FE_OVERFLOW);                                                                          \
      std::feclearexcept(FE_UNDERFLOW);                                                                         \
      auto result  = x op y;                                                                                    \
      if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) {                                  \
        throw std::runtime_error("overflow / underflow");                                                       \
      }                                                                                                         \
      return result;                                                                                            \
    });                                                                                                         \
}

DEF_OPERATOR(ExpressionRange::operator+, +)
DEF_OPERATOR(ExpressionRange::operator-, -)
DEF_OPERATOR(ExpressionRange::operator*, *)

ExpressionRange ExpressionRange::operator/(const ExpressionRange& other) const {
  if (type != ExpressionRangeType::Integer || other.type != ExpressionRangeType::Integer) {
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  }
  if (other.int_min * other.int_max <= 0) {
    // if the other interval contains 0, the rule is more complicated;
    // punt for now, we can revisit by splitting the other interval and
    // taking the convex hull of the resulting two intervals
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  }
  return binOp<int64_t>(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) / y); });
}

ExpressionRange ExpressionRange::operator||(const ExpressionRange& other) const {
  if (type != other.type) {
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  }
  ExpressionRange result;
  switch (type) {
  case ExpressionRangeType::Invalid:
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  case ExpressionRangeType::Integer: {
    result.type = ExpressionRangeType::Integer;
    result.int_min = std::min(int_min, other.int_min);
    result.int_max = std::max(int_max, other.int_max);
    break;
  }
  case ExpressionRangeType::FloatingPoint: {
    result.type = ExpressionRangeType::FloatingPoint;
    result.fp_min = std::min(fp_min, other.fp_min);
    result.fp_max = std::max(fp_max, other.fp_max);
    break;
  }
  default:
    CHECK(false);
  }
  return result;
}

ExpressionRange getExpressionRange(
    const Analyzer::BinOper* expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments);

ExpressionRange getExpressionRange(const Analyzer::Constant* expr);

ExpressionRange getExpressionRange(
    const Analyzer::ColumnVar* col_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments);

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr);

ExpressionRange getExpressionRange(
    const Analyzer::CaseExpr* case_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments);

ExpressionRange getExpressionRange(
    const Analyzer::UOper* u_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments);

ExpressionRange getExpressionRange(
    const Analyzer::ExtractExpr* extract_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments);

ExpressionRange getExpressionRange(
    const Analyzer::Expr* expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  auto bin_oper_expr = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper_expr) {
    return getExpressionRange(bin_oper_expr, fragments);
  }
  auto constant_expr = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant_expr) {
    return getExpressionRange(constant_expr);
  }
  auto column_var_expr = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (column_var_expr) {
    return getExpressionRange(column_var_expr, fragments);
  }
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    return getExpressionRange(like_expr);
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return getExpressionRange(case_expr, fragments);
  }
  auto u_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_expr) {
    return getExpressionRange(u_expr, fragments);
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return getExpressionRange(extract_expr, fragments);
  }
  return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
}

ExpressionRange getExpressionRange(
    const Analyzer::BinOper* expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const auto& lhs = getExpressionRange(expr->get_left_operand(), fragments);
  const auto& rhs = getExpressionRange(expr->get_right_operand(), fragments);
  switch (expr->get_optype()) {
  case kPLUS:
    return lhs + rhs;
  case kMINUS:
    return lhs - rhs;
  case kMULTIPLY:
    return lhs * rhs;
  case kDIVIDE:
    return lhs / rhs;
  default:
    break;
  }
  return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
}

ExpressionRange getExpressionRange(const Analyzer::Constant* constant_expr) {
  if (constant_expr->get_is_null()) {
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  }
  const auto constant_type = constant_expr->get_type_info().get_type();
  switch (constant_type) {
  case kSMALLINT: {
    const auto v = constant_expr->get_constval().smallintval;
    return { ExpressionRangeType::Integer, { v }, { v } };
  }
  case kINT: {
    const auto v = constant_expr->get_constval().intval;
    return { ExpressionRangeType::Integer, { v }, { v } };
  }
  case kBIGINT: {
    const auto v = constant_expr->get_constval().bigintval;
    return { ExpressionRangeType::Integer, { v }, { v } };
  }
  case kTIME:
  case kTIMESTAMP:
  case kDATE: {
    const auto v = constant_expr->get_constval().timeval;
    return { ExpressionRangeType::Integer, { v }, { v } };
  }
  case kDOUBLE: {
    const auto v = constant_expr->get_constval().doubleval;
    ExpressionRange result;
    result.type = ExpressionRangeType::FloatingPoint;
    result.fp_min = v;
    result.fp_max = v;
    return result;
  }
  default:
    break;
  }
  return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
}

#define FIND_STAT_FRAG(stat_name)                                                                  \
  const auto stat_name##_frag = std::stat_name##_element(fragments.begin(), fragments.end(),       \
    [&has_nulls, col_id, col_ti](const Fragmenter_Namespace::FragmentInfo& lhs,                    \
                                 const Fragmenter_Namespace::FragmentInfo& rhs) {                  \
      auto lhs_meta_it = lhs.chunkMetadataMap.find(col_id);                                        \
      CHECK(lhs_meta_it != lhs.chunkMetadataMap.end());                                            \
      auto rhs_meta_it = rhs.chunkMetadataMap.find(col_id);                                        \
      CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());                                            \
      if (lhs_meta_it->second.chunkStats.has_nulls || rhs_meta_it->second.chunkStats.has_nulls) {  \
        has_nulls = true;                                                                          \
      }                                                                                            \
      return extract_##stat_name##_stat(lhs_meta_it->second.chunkStats, col_ti) <                  \
             extract_##stat_name##_stat(rhs_meta_it->second.chunkStats, col_ti);                   \
  });                                                                                              \
  if (has_nulls || stat_name##_frag == fragments.end()) {                                          \
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };                                         \
  }

namespace {

inline double extract_min_stat_double(const ChunkStats& stats) {
  if (stats.has_nulls) {  // clobber the additional information for now
    return NULL_DOUBLE;
  }
  return stats.min.doubleval;
}

inline double extract_max_stat_double(const ChunkStats& stats) {
  return stats.max.doubleval;
}

}  // namespace

ExpressionRange getExpressionRange(
    const Analyzer::ColumnVar* col_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  int col_id = col_expr->get_column_id();
  const auto& col_ti = col_expr->get_type_info();
  switch (col_ti.get_type()) {
  case kTEXT:
  case kCHAR:
  case kVARCHAR:
    CHECK_EQ(kENCODING_DICT, col_ti.get_compression());
  case kBOOLEAN:
  case kSMALLINT:
  case kINT:
  case kBIGINT:
  case kDATE:
  case kTIMESTAMP:
  case kTIME:
  case kDOUBLE: {
    bool has_nulls { false };
    FIND_STAT_FRAG(min);
    FIND_STAT_FRAG(max);
    const auto min_it = min_frag->chunkMetadataMap.find(col_id);
    CHECK(min_it != min_frag->chunkMetadataMap.end());
    const auto max_it = max_frag->chunkMetadataMap.find(col_id);
    CHECK(max_it != max_frag->chunkMetadataMap.end());
    if (col_ti.get_type() == kDOUBLE) {
      ExpressionRange result;
      result.type = ExpressionRangeType::FloatingPoint;
      result.fp_min = extract_min_stat_double(min_it->second.chunkStats);
      result.fp_max = extract_max_stat_double(max_it->second.chunkStats);
      return result;
    }
    const auto min_val = extract_min_stat(min_it->second.chunkStats, col_ti);
    const auto max_val = extract_max_stat(max_it->second.chunkStats, col_ti);
    CHECK_GE(max_val, min_val);
    return { ExpressionRangeType::Integer, { min_val }, { max_val } };
  }
  default:
    break;
  }
  return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
}

#undef FIND_STAT_FRAG

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr) {
  const auto& ti = like_expr->get_type_info();
  CHECK(ti.is_boolean());
  const auto& arg_ti = like_expr->get_arg()->get_type_info();
  return { ExpressionRangeType::Integer, { arg_ti.get_notnull() ? 0 : inline_int_null_val(ti) }, { 1 } };
}

ExpressionRange getExpressionRange(
    const Analyzer::CaseExpr* case_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  ExpressionRange expr_range { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  const auto& case_ti = case_expr->get_type_info();
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().get_type(), kBOOLEAN);
    CHECK(expr_pair.second->get_type_info() == case_ti);
    const auto crt_range = getExpressionRange(expr_pair.second, fragments);
    if (crt_range.type == ExpressionRangeType::Invalid) {
      return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
    }
    expr_range = (expr_range.type != ExpressionRangeType::Invalid) ? expr_range || crt_range : crt_range;
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  return expr_range || getExpressionRange(else_expr, fragments);
}

ExpressionRange getExpressionRange(
    const Analyzer::UOper* u_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  if (u_expr->get_optype() != kCAST) {
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  }
  const auto arg_range = getExpressionRange(u_expr->get_operand(), fragments);
  switch (arg_range.type) {
  case ExpressionRangeType::FloatingPoint: {
    if (u_expr->get_type_info().is_integer()) {
      ExpressionRange result;
      result.type = ExpressionRangeType::Integer;
      result.int_min = arg_range.fp_min;
      result.int_max = arg_range.fp_max;
      return result;
    }
    break;
  }
  case ExpressionRangeType::Integer: {
    if (u_expr->get_type_info().is_integer() || u_expr->get_type_info().is_time()) {
      return arg_range;
    }
    if (u_expr->get_type_info().get_type() == kDOUBLE) {
      ExpressionRange result;
      result.type = ExpressionRangeType::Integer;
      result.fp_min = arg_range.int_min;
      result.fp_max = arg_range.int_max;
      return result;
    }
    break;
  }
  case ExpressionRangeType::Invalid:
    break;
  default:
    CHECK(false);
  }
  return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
}

ExpressionRange getExpressionRange(
    const Analyzer::ExtractExpr* extract_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const int32_t extract_field { extract_expr->get_field() };
  ExpressionRange result;
  result.type = ExpressionRangeType::Integer;
  switch (extract_field) {
  case kYEAR:
    return { ExpressionRangeType::Invalid, { 0 }, { 0 } };
  case kEPOCH:
    return getExpressionRange(extract_expr->get_from_expr(), fragments);
  case kMONTH:
    result.int_min = 1;
    result.int_max = 12;
    break;
  case kDAY:
    result.int_min = 1;
    result.int_max = 31;
    break;
  case kHOUR:
    result.int_min = 0;
    result.int_max = 23;
    break;
  case kMINUTE:
    result.int_min = 0;
    result.int_max = 59;
    break;
  case kSECOND:
    result.int_min = 0;
    result.int_max = 60;
    break;
  case kDOW:
    result.int_min = 0;
    result.int_max = 6;
    break;
  case kDOY:
    result.int_min = 1;
    result.int_max = 366;
    break;
  default:
    CHECK(false);
  }
  return result;
}
