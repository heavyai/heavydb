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

#include "WindowExpressionRewrite.h"

namespace {

// Returns true iff the case expression has an else null branch.
bool matches_else_null(const hdk::ir::CaseExpr* case_expr) {
  const auto else_null =
      dynamic_cast<const hdk::ir::Constant*>(case_expr->get_else_expr());
  return else_null && else_null->get_is_null();
}

// Returns true iff the expression is a big integer greater than 0.
bool matches_gt_bigint_zero(const hdk::ir::BinOper* window_gt_zero) {
  if (window_gt_zero->get_optype() != kGT) {
    return false;
  }
  const auto zero =
      dynamic_cast<const hdk::ir::Constant*>(window_gt_zero->get_right_operand());
  return zero && zero->get_type_info().get_type() == kBIGINT &&
         zero->get_constval().bigintval == 0;
}

// Returns true iff the sum and the count match in type and arguments. Used to replace
// combination can be replaced with an explicit average.
bool window_sum_and_count_match(const hdk::ir::WindowFunction* sum_window_expr,
                                const hdk::ir::WindowFunction* count_window_expr) {
  CHECK_EQ(count_window_expr->get_type_info().get_type(), kBIGINT);
  return expr_list_match(sum_window_expr->getArgs(), count_window_expr->getArgs());
}

bool is_sum_kind(const SqlWindowFunctionKind kind) {
  return kind == SqlWindowFunctionKind::SUM_INTERNAL ||
         kind == SqlWindowFunctionKind::SUM;
}

}  // namespace

std::shared_ptr<hdk::ir::WindowFunction> rewrite_sum_window(const hdk::ir::Expr* expr) {
  const auto case_expr = dynamic_cast<const hdk::ir::CaseExpr*>(expr);
  if (!case_expr || !matches_else_null(case_expr)) {
    return nullptr;
  }
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  if (expr_pair_list.size() != 1) {
    return nullptr;
  }
  const auto& expr_pair = expr_pair_list.front();
  const auto window_gt_zero =
      dynamic_cast<const hdk::ir::BinOper*>(expr_pair.first.get());
  if (!window_gt_zero || !matches_gt_bigint_zero(window_gt_zero)) {
    return nullptr;
  }
  const auto sum_window_expr =
      std::dynamic_pointer_cast<hdk::ir::WindowFunction>(remove_cast(expr_pair.second));
  if (!sum_window_expr || !is_sum_kind(sum_window_expr->getKind())) {
    return nullptr;
  }
  const auto count_window_expr = std::dynamic_pointer_cast<const hdk::ir::WindowFunction>(
      remove_cast(window_gt_zero->get_own_left_operand()));
  if (!count_window_expr ||
      count_window_expr->getKind() != SqlWindowFunctionKind::COUNT) {
    return nullptr;
  }
  if (!window_sum_and_count_match(sum_window_expr.get(), count_window_expr.get())) {
    return nullptr;
  }
  CHECK(sum_window_expr);
  auto sum_ti = sum_window_expr->get_type_info();
  if (sum_ti.is_integer()) {
    sum_ti = SQLTypeInfo(kBIGINT, sum_ti.get_notnull());
  }
  return hdk::ir::makeExpr<hdk::ir::WindowFunction>(sum_ti,
                                                    SqlWindowFunctionKind::SUM,
                                                    sum_window_expr->getArgs(),
                                                    sum_window_expr->getPartitionKeys(),
                                                    sum_window_expr->getOrderKeys(),
                                                    sum_window_expr->getCollation());
}

std::shared_ptr<hdk::ir::WindowFunction> rewrite_avg_window(const hdk::ir::Expr* expr) {
  const auto cast_expr = dynamic_cast<const hdk::ir::UOper*>(expr);
  const auto div_expr = dynamic_cast<const hdk::ir::BinOper*>(
      cast_expr && cast_expr->get_optype() == kCAST ? cast_expr->get_operand() : expr);
  if (!div_expr || div_expr->get_optype() != kDIVIDE) {
    return nullptr;
  }
  const auto sum_window_expr = rewrite_sum_window(div_expr->get_left_operand());
  if (!sum_window_expr) {
    return nullptr;
  }
  const auto cast_count_window =
      dynamic_cast<const hdk::ir::UOper*>(div_expr->get_right_operand());
  if (cast_count_window && cast_count_window->get_optype() != kCAST) {
    return nullptr;
  }
  const auto count_window = dynamic_cast<const hdk::ir::WindowFunction*>(
      cast_count_window ? cast_count_window->get_operand()
                        : div_expr->get_right_operand());
  if (!count_window || count_window->getKind() != SqlWindowFunctionKind::COUNT) {
    return nullptr;
  }
  CHECK_EQ(count_window->get_type_info().get_type(), kBIGINT);
  if (cast_count_window && cast_count_window->get_type_info().get_type() !=
                               sum_window_expr->get_type_info().get_type()) {
    return nullptr;
  }
  if (!expr_list_match(sum_window_expr.get()->getArgs(), count_window->getArgs())) {
    return nullptr;
  }
  return hdk::ir::makeExpr<hdk::ir::WindowFunction>(SQLTypeInfo(kDOUBLE),
                                                    SqlWindowFunctionKind::AVG,
                                                    sum_window_expr->getArgs(),
                                                    sum_window_expr->getPartitionKeys(),
                                                    sum_window_expr->getOrderKeys(),
                                                    sum_window_expr->getCollation());
}
