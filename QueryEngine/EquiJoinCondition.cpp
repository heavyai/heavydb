/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/EquiJoinCondition.h"

#include "Analyzer/Analyzer.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/RangeTableIndexVisitor.h"

#include <optional>

namespace {

struct NormalizedEquiJoinQual {
  std::shared_ptr<Analyzer::Expr> qual;
  std::set<int> outer_rte_set;
  shared::TableKey inner_table_key;
  int inner_rte_idx;
  SQLOps op_type;
};

std::shared_ptr<Analyzer::Expr> remove_safe_join_normalization_cast(
    const std::shared_ptr<Analyzer::Expr>& expr) {
  const auto uoper = std::dynamic_pointer_cast<Analyzer::UOper>(expr);
  if (!uoper || uoper->get_optype() != kCAST) {
    return expr;
  }
  const auto& operand_ti = uoper->get_operand()->get_type_info();
  const auto& cast_ti = uoper->get_type_info();
  if (operand_ti.is_decimal() || cast_ti.is_decimal()) {
    return expr;
  }
  return uoper->get_own_operand();
}

std::shared_ptr<Analyzer::ColumnVar> get_join_side_col_var(
    const std::shared_ptr<Analyzer::Expr>& expr) {
  auto col_var = std::dynamic_pointer_cast<Analyzer::ColumnVar>(
      remove_safe_join_normalization_cast(expr));
  if (col_var) {
    return col_var;
  }
  const auto string_oper = std::dynamic_pointer_cast<Analyzer::StringOper>(
      remove_safe_join_normalization_cast(expr));
  if (string_oper && string_oper->getArity() >= 1UL) {
    return std::dynamic_pointer_cast<Analyzer::ColumnVar>(string_oper->getOwnArg(0));
  }
  return nullptr;
}

std::optional<NormalizedEquiJoinQual> normalize_equi_join_qual(
    const std::shared_ptr<Analyzer::Expr>& qual) {
  const auto bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(qual);
  if (!bin_oper || !IS_EQUIVALENCE(bin_oper->get_optype()) ||
      bin_oper->get_qualifier() != kONE) {
    return std::nullopt;
  }

  auto lhs = remove_safe_join_normalization_cast(bin_oper->get_own_left_operand());
  auto rhs = remove_safe_join_normalization_cast(bin_oper->get_own_right_operand());
  const auto lhs_col = get_join_side_col_var(lhs);
  const auto rhs_col = get_join_side_col_var(rhs);
  if (!lhs_col && !rhs_col) {
    return std::nullopt;
  }

  AllRangeTableIndexVisitor visitor;
  const auto lhs_rte_set = visitor.visit(lhs.get());
  const auto rhs_rte_set = visitor.visit(rhs.get());
  if (lhs_rte_set.size() != 1 || rhs_rte_set.size() != 1 || lhs_rte_set == rhs_rte_set) {
    return std::nullopt;
  }

  const auto lhs_max_rte = *lhs_rte_set.rbegin();
  const auto rhs_max_rte = *rhs_rte_set.rbegin();
  const bool lhs_is_inner =
      lhs_col && (!rhs_col || lhs_col->get_rte_idx() > rhs_col->get_rte_idx() ||
                  (!rhs_col && lhs_max_rte > rhs_max_rte));
  const auto inner_expr = lhs_is_inner ? lhs : rhs;
  const auto outer_expr = lhs_is_inner ? rhs : lhs;
  const auto inner_col = lhs_is_inner ? lhs_col : rhs_col;
  const auto outer_rte_set = lhs_is_inner ? rhs_rte_set : lhs_rte_set;
  if (!inner_col) {
    return std::nullopt;
  }

  auto normalized_qual = std::make_shared<Analyzer::BinOper>(bin_oper->get_type_info(),
                                                             false,
                                                             bin_oper->get_optype(),
                                                             bin_oper->get_qualifier(),
                                                             outer_expr,
                                                             inner_expr);
  return NormalizedEquiJoinQual{normalized_qual,
                                outer_rte_set,
                                inner_col->getTableKey(),
                                inner_col->get_rte_idx(),
                                bin_oper->get_optype()};
}

// Returns true iff crt and prev are both equi-join conditions on the same pair of
// left-deep inputs after orienting the newer input as the inner side.
bool can_combine_with(const NormalizedEquiJoinQual& crt,
                      const NormalizedEquiJoinQual& prev) {
  // We could accept a mix of kEQ and kBW_EQ, but don't bother for now.
  return crt.op_type == prev.op_type && crt.outer_rte_set == prev.outer_rte_set &&
         crt.inner_table_key == prev.inner_table_key &&
         crt.inner_rte_idx == prev.inner_rte_idx;
}

std::list<std::shared_ptr<Analyzer::Expr>> make_composite_equals_impl(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& crt_coalesced_quals) {
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
  std::vector<std::shared_ptr<Analyzer::Expr>> lhs_tuple;
  std::vector<std::shared_ptr<Analyzer::Expr>> rhs_tuple;
  bool not_null{true};
  for (const auto& qual : crt_coalesced_quals) {
    const auto qual_binary = std::dynamic_pointer_cast<Analyzer::BinOper>(qual);
    CHECK(qual_binary);
    not_null = not_null && qual_binary->get_type_info().get_notnull();
    const auto lhs_col =
        remove_safe_join_normalization_cast(qual_binary->get_own_left_operand());
    const auto rhs_col =
        remove_safe_join_normalization_cast(qual_binary->get_own_right_operand());
    const auto lhs_ti = lhs_col->get_type_info();
    // Coalesce cols for integers, bool, and dict encoded strings. Forces baseline hash
    // join.
    if (IS_NUMBER(lhs_ti.get_type()) ||
        (IS_STRING(lhs_ti.get_type()) && lhs_ti.get_compression() == kENCODING_DICT) ||
        (lhs_ti.get_type() == kBOOLEAN)) {
      lhs_tuple.push_back(lhs_col);
      rhs_tuple.push_back(rhs_col);
    } else {
      join_quals.push_back(qual);
    }
  }
  CHECK(!crt_coalesced_quals.empty());
  const auto first_qual =
      std::dynamic_pointer_cast<Analyzer::BinOper>(crt_coalesced_quals.front());
  CHECK(first_qual);
  CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
  if (lhs_tuple.size() > 0) {
    join_quals.push_front(std::make_shared<Analyzer::BinOper>(
        SQLTypeInfo(kBOOLEAN, not_null),
        false,
        first_qual->get_optype(),
        kONE,
        lhs_tuple.size() > 1 ? std::make_shared<Analyzer::ExpressionTuple>(lhs_tuple)
                             : lhs_tuple.front(),
        rhs_tuple.size() > 1 ? std::make_shared<Analyzer::ExpressionTuple>(rhs_tuple)
                             : rhs_tuple.front()));
  }
  return join_quals;
}

// Create an equals expression with column tuple operands out of regular equals
// expressions.
std::list<std::shared_ptr<Analyzer::Expr>> make_composite_equals(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& crt_coalesced_quals) {
  if (crt_coalesced_quals.size() == 1) {
    return {crt_coalesced_quals.front()};
  }
  return make_composite_equals_impl(crt_coalesced_quals);
}

}  // namespace

std::list<std::shared_ptr<Analyzer::Expr>> combine_equi_join_conditions(
    const std::list<std::shared_ptr<Analyzer::Expr>>& join_quals) {
  if (join_quals.empty()) {
    return {};
  }
  std::list<std::shared_ptr<Analyzer::Expr>> coalesced_quals;
  std::vector<std::shared_ptr<Analyzer::Expr>> crt_coalesced_quals;
  std::optional<NormalizedEquiJoinQual> prev_normalized_qual;
  auto flush_current = [&]() {
    if (!crt_coalesced_quals.empty()) {
      coalesced_quals.splice(coalesced_quals.end(),
                             make_composite_equals(crt_coalesced_quals));
      crt_coalesced_quals.clear();
      prev_normalized_qual = std::nullopt;
    }
  };
  for (const auto& simple_join_qual : join_quals) {
    auto normalized_qual = normalize_equi_join_qual(simple_join_qual);
    if (!normalized_qual) {
      flush_current();
      coalesced_quals.push_back(simple_join_qual);
      continue;
    }
    if (crt_coalesced_quals.size() >= g_maximum_conditions_to_coalesce ||
        (prev_normalized_qual &&
         !can_combine_with(*normalized_qual, *prev_normalized_qual))) {
      flush_current();
    }
    crt_coalesced_quals.push_back(normalized_qual->qual);
    prev_normalized_qual = std::move(normalized_qual);
  }
  flush_current();
  return coalesced_quals;
}

std::list<std::shared_ptr<Analyzer::Expr>> coalesce_singleton_equi_join(
    const std::shared_ptr<Analyzer::BinOper>& join_qual) {
  std::vector<std::shared_ptr<Analyzer::Expr>> singleton_qual_list;
  singleton_qual_list.push_back(join_qual);
  return make_composite_equals_impl(singleton_qual_list);
}
