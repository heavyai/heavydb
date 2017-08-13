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
#include "EquiJoinCondition.h"
#include "HashJoinRuntime.h"
#include "Analyzer/Analyzer.h"

namespace {

std::shared_ptr<Analyzer::Expr> remove_cast(const std::shared_ptr<Analyzer::Expr>& expr) {
  const auto uoper = dynamic_cast<const Analyzer::UOper*>(expr.get());
  if (!uoper || uoper->get_optype() != kCAST) {
    return expr;
  }
  return uoper->get_own_operand();
}

// Returns true iff crt and prev are both equi-join conditions on the same pair of tables.
bool can_combine_with(const Analyzer::Expr* crt, const Analyzer::Expr* prev) {
  const auto crt_bin = dynamic_cast<const Analyzer::BinOper*>(crt);
  const auto prev_bin = dynamic_cast<const Analyzer::BinOper*>(prev);
  if (!crt_bin || !prev_bin) {
    return false;
  }
  if (crt_bin->get_optype() != kEQ || prev_bin->get_optype() != kEQ) {
    return false;
  }
  const auto crt_outer = std::dynamic_pointer_cast<Analyzer::ColumnVar>(remove_cast(crt_bin->get_own_left_operand()));
  const auto prev_outer = std::dynamic_pointer_cast<Analyzer::ColumnVar>(remove_cast(prev_bin->get_own_left_operand()));
  if (!crt_outer || !prev_outer || crt_outer->get_table_id() != prev_outer->get_table_id()) {
    return false;
  }
  const auto crt_inner = std::dynamic_pointer_cast<Analyzer::ColumnVar>(remove_cast(crt_bin->get_own_right_operand()));
  const auto prev_inner =
      std::dynamic_pointer_cast<Analyzer::ColumnVar>(remove_cast(prev_bin->get_own_right_operand()));
  if (!crt_inner || !prev_inner || crt_inner->get_table_id() != prev_inner->get_table_id()) {
    return false;
  }
  return true;
}

std::shared_ptr<Analyzer::BinOper> make_composite_equals_impl(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& crt_coalesced_quals) {
  std::vector<std::shared_ptr<Analyzer::ColumnVar>> lhs_tuple;
  std::vector<std::shared_ptr<Analyzer::ColumnVar>> rhs_tuple;
  bool not_null{true};
  for (const auto& qual : crt_coalesced_quals) {
    const auto qual_binary = std::dynamic_pointer_cast<Analyzer::BinOper>(qual);
    CHECK(qual_binary);
    not_null = not_null && qual_binary->get_type_info().get_notnull();
    const auto lhs_col =
        std::dynamic_pointer_cast<Analyzer::ColumnVar>(remove_cast(qual_binary->get_own_left_operand()));
    const auto rhs_col =
        std::dynamic_pointer_cast<Analyzer::ColumnVar>(remove_cast(qual_binary->get_own_right_operand()));
    CHECK(lhs_col && rhs_col);
    lhs_tuple.push_back(lhs_col);
    rhs_tuple.push_back(rhs_col);
  }
  return std::make_shared<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, not_null),
                                             false,
                                             kEQ,
                                             kONE,
                                             std::make_shared<Analyzer::ColumnVarTuple>(lhs_tuple),
                                             std::make_shared<Analyzer::ColumnVarTuple>(rhs_tuple));
}

// Create an equals expression with column tuple operands out of regular equals expressions.
std::shared_ptr<Analyzer::Expr> make_composite_equals(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& crt_coalesced_quals) {
  if (crt_coalesced_quals.size() == 1) {
    return crt_coalesced_quals.front();
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
  for (const auto& simple_join_qual : join_quals) {
    if (crt_coalesced_quals.empty()) {
      crt_coalesced_quals.push_back(simple_join_qual);
      continue;
    }
    if (crt_coalesced_quals.size() >= g_maximum_conditions_to_coalesce ||
        !can_combine_with(simple_join_qual.get(), crt_coalesced_quals.back().get())) {
      coalesced_quals.push_back(make_composite_equals(crt_coalesced_quals));
      crt_coalesced_quals.clear();
    }
    crt_coalesced_quals.push_back(simple_join_qual);
  }
  if (!crt_coalesced_quals.empty()) {
    coalesced_quals.push_back(make_composite_equals(crt_coalesced_quals));
  }
  return coalesced_quals;
}

std::shared_ptr<Analyzer::BinOper> coalesce_singleton_equi_join(const std::shared_ptr<Analyzer::BinOper>& join_qual) {
  std::vector<std::shared_ptr<Analyzer::Expr>> singleton_qual_list;
  singleton_qual_list.push_back(join_qual);
  return make_composite_equals_impl(singleton_qual_list);
}
