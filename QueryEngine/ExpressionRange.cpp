#include "ExpressionRange.h"
#include "GroupByAndAggregate.h"


ExpressionRange ExpressionRange::operator+(const ExpressionRange& other) const {
  return binOp(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) + y); });
}

ExpressionRange ExpressionRange::operator-(const ExpressionRange& other) const {
  return binOp(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) - y); });
}

ExpressionRange ExpressionRange::operator*(const ExpressionRange& other) const {
  return binOp(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) * y); });
}

ExpressionRange ExpressionRange::operator/(const ExpressionRange& other) const {
  if (other.min * other.max <= 0) {
    // if the other interval contains 0, the rule is more complicated;
    // punt for now, we can revisit by splitting the other interval and
    // taking the convex hull of the resulting two intervals
    return { false, 0, 0 };
  }
  return binOp(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) / y); });
}

ExpressionRange ExpressionRange::operator||(const ExpressionRange& other) const {
  if (!valid || !other.valid) {
    return { false, 0, 0 };
  }
  return { true, std::min(min, other.min), std::max(max, other.max) };
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
  return { false, 0, 0 };
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
  return { false, 0, 0 };
}

ExpressionRange getExpressionRange(const Analyzer::Constant* constant_expr) {
  if (constant_expr->get_is_null()) {
    return { false, 0, 0 };
  }
  const auto constant_type = constant_expr->get_type_info().get_type();
  switch (constant_type) {
  case kSMALLINT: {
    const auto v = constant_expr->get_constval().smallintval;
    return { true, v, v };
  }
  case kINT: {
    const auto v = constant_expr->get_constval().intval;
    return { true, v, v };
  }
  case kBIGINT: {
    const auto v = constant_expr->get_constval().bigintval;
    return { true, v, v };
  }
  case kTIME:
  case kTIMESTAMP:
  case kDATE: {
    const auto v = constant_expr->get_constval().timeval;
    return { true, v, v };
  }
  default:
    break;
  }
  return { false, 0, 0 };
}

#define FIND_STAT_FRAG(stat_name)                                                             \
  const auto stat_name##_frag = std::stat_name##_element(fragments.begin(), fragments.end(),  \
    [col_id, col_ti](const Fragmenter_Namespace::FragmentInfo& lhs,                           \
                     const Fragmenter_Namespace::FragmentInfo& rhs) {                         \
      auto lhs_meta_it = lhs.chunkMetadataMap.find(col_id);                                   \
      CHECK(lhs_meta_it != lhs.chunkMetadataMap.end());                                       \
      auto rhs_meta_it = rhs.chunkMetadataMap.find(col_id);                                   \
      CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());                                       \
      return extract_##stat_name##_stat(lhs_meta_it->second.chunkStats, col_ti) <             \
             extract_##stat_name##_stat(rhs_meta_it->second.chunkStats, col_ti);              \
  });                                                                                         \
  if (stat_name##_frag == fragments.end()) {                                                  \
    return { false, 0, 0 };                                                                   \
  }

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
  case kBIGINT: {
    FIND_STAT_FRAG(min);
    FIND_STAT_FRAG(max);
    const auto min_it = min_frag->chunkMetadataMap.find(col_id);
    CHECK(min_it != min_frag->chunkMetadataMap.end());
    const auto max_it = max_frag->chunkMetadataMap.find(col_id);
    CHECK(max_it != max_frag->chunkMetadataMap.end());
    const auto min_val = extract_min_stat(min_it->second.chunkStats, col_ti);
    const auto max_val = extract_max_stat(max_it->second.chunkStats, col_ti);
    CHECK_GE(max_val, min_val);
    return { true, min_val, max_val };
  }
  case kFLOAT:
  case kDOUBLE:
    return { false, 0, 0 };
  default:
    break;
  }
  return { false, 0, 0 };
}

#undef FIND_STAT_FRAG

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr) {
  const auto& ti = like_expr->get_type_info();
  CHECK(ti.is_boolean());
  const auto& arg_ti = like_expr->get_arg()->get_type_info();
  return { true, arg_ti.get_notnull() ? 0 : inline_int_null_val(ti), 1 };
}

ExpressionRange getExpressionRange(
    const Analyzer::CaseExpr* case_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  ExpressionRange expr_range { false, 0, 0 };
  const auto& case_ti = case_expr->get_type_info();
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().get_type(), kBOOLEAN);
    CHECK(expr_pair.second->get_type_info() == case_ti);
    const auto crt_range = getExpressionRange(expr_pair.second, fragments);
    expr_range = expr_range.valid ? expr_range || crt_range : crt_range;
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  return expr_range || getExpressionRange(else_expr, fragments);
}
