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

#include "ScalarExprVisitor.h"

class DeepCopyVisitor : public ScalarExprVisitor<hdk::ir::ExprPtr> {
 protected:
  using RetType = hdk::ir::ExprPtr;
  RetType visitColumnVar(const hdk::ir::ColumnVar* col_var) const override {
    return col_var->deep_copy();
  }

  RetType visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    return col_ref->deep_copy();
  }

  RetType visitGroupColumnRef(const hdk::ir::GroupColumnRef* col_ref) const override {
    return col_ref->deep_copy();
  }

  RetType visitColumnVarTuple(
      const hdk::ir::ExpressionTuple* col_var_tuple) const override {
    return col_var_tuple->deep_copy();
  }

  RetType visitVar(const hdk::ir::Var* var) const override { return var->deep_copy(); }

  RetType visitConstant(const hdk::ir::Constant* constant) const override {
    return constant->deep_copy();
  }

  RetType visitUOper(const hdk::ir::UOper* uoper) const override {
    return hdk::ir::makeExpr<hdk::ir::UOper>(uoper->get_type_info(),
                                             uoper->get_contains_agg(),
                                             uoper->get_optype(),
                                             visit(uoper->get_operand()));
  }

  RetType visitBinOper(const hdk::ir::BinOper* bin_oper) const override {
    return hdk::ir::makeExpr<hdk::ir::BinOper>(bin_oper->get_type_info(),
                                               bin_oper->get_contains_agg(),
                                               bin_oper->get_optype(),
                                               bin_oper->get_qualifier(),
                                               visit(bin_oper->get_left_operand()),
                                               visit(bin_oper->get_right_operand()));
  }

  RetType visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    return subquery->deep_copy();
  }

  RetType visitInValues(const hdk::ir::InValues* in_values) const override {
    const auto& value_list = in_values->get_value_list();
    std::list<RetType> new_list;
    for (const auto& in_value : value_list) {
      new_list.push_back(visit(in_value.get()));
    }
    return hdk::ir::makeExpr<hdk::ir::InValues>(visit(in_values->get_arg()), new_list);
  }

  RetType visitInIntegerSet(const hdk::ir::InIntegerSet* in_integer_set) const override {
    return hdk::ir::makeExpr<hdk::ir::InIntegerSet>(
        visit(in_integer_set->get_arg()),
        in_integer_set->get_value_list(),
        in_integer_set->get_type_info().get_notnull());
  }

  RetType visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    return hdk::ir::makeExpr<hdk::ir::InSubquery>(in_subquery->get_type_info(),
                                                  visit(in_subquery->getArg().get()),
                                                  in_subquery->getNodeShared());
  }

  RetType visitCharLength(const hdk::ir::CharLengthExpr* char_length) const override {
    return hdk::ir::makeExpr<hdk::ir::CharLengthExpr>(
        visit(char_length->get_arg()), char_length->get_calc_encoded_length());
  }

  RetType visitKeyForString(const hdk::ir::KeyForStringExpr* expr) const override {
    return hdk::ir::makeExpr<hdk::ir::KeyForStringExpr>(visit(expr->get_arg()));
  }

  RetType visitSampleRatio(const hdk::ir::SampleRatioExpr* expr) const override {
    return hdk::ir::makeExpr<hdk::ir::SampleRatioExpr>(visit(expr->get_arg()));
  }

  RetType visitLower(const hdk::ir::LowerExpr* expr) const override {
    return hdk::ir::makeExpr<hdk::ir::LowerExpr>(visit(expr->get_arg()));
  }

  RetType visitCardinality(const hdk::ir::CardinalityExpr* cardinality) const override {
    return hdk::ir::makeExpr<hdk::ir::CardinalityExpr>(visit(cardinality->get_arg()));
  }

  RetType visitLikeExpr(const hdk::ir::LikeExpr* like) const override {
    auto escape_expr = like->get_escape_expr();
    return hdk::ir::makeExpr<hdk::ir::LikeExpr>(
        visit(like->get_arg()),
        visit(like->get_like_expr()),
        escape_expr ? visit(escape_expr) : nullptr,
        like->get_is_ilike(),
        like->get_is_simple());
  }

  RetType visitRegexpExpr(const hdk::ir::RegexpExpr* regexp) const override {
    auto escape_expr = regexp->get_escape_expr();
    return hdk::ir::makeExpr<hdk::ir::RegexpExpr>(
        visit(regexp->get_arg()),
        visit(regexp->get_pattern_expr()),
        escape_expr ? visit(escape_expr) : nullptr);
  }

  RetType visitWidthBucket(
      const hdk::ir::WidthBucketExpr* width_bucket_expr) const override {
    return hdk::ir::makeExpr<hdk::ir::WidthBucketExpr>(
        visit(width_bucket_expr->get_target_value()),
        visit(width_bucket_expr->get_lower_bound()),
        visit(width_bucket_expr->get_upper_bound()),
        visit(width_bucket_expr->get_partition_count()));
  }

  RetType visitCaseExpr(const hdk::ir::CaseExpr* case_expr) const override {
    std::list<std::pair<RetType, RetType>> new_list;
    for (auto p : case_expr->get_expr_pair_list()) {
      new_list.emplace_back(visit(p.first.get()), visit(p.second.get()));
    }
    auto else_expr = case_expr->get_else_expr();
    return hdk::ir::makeExpr<hdk::ir::CaseExpr>(
        case_expr->get_type_info(),
        case_expr->get_contains_agg(),
        new_list,
        else_expr == nullptr ? nullptr : visit(else_expr));
  }

  RetType visitDatetruncExpr(const hdk::ir::DatetruncExpr* datetrunc) const override {
    return hdk::ir::makeExpr<hdk::ir::DatetruncExpr>(datetrunc->get_type_info(),
                                                     datetrunc->get_contains_agg(),
                                                     datetrunc->get_field(),
                                                     visit(datetrunc->get_from_expr()));
  }

  RetType visitExtractExpr(const hdk::ir::ExtractExpr* extract) const override {
    return hdk::ir::makeExpr<hdk::ir::ExtractExpr>(extract->get_type_info(),
                                                   extract->get_contains_agg(),
                                                   extract->get_field(),
                                                   visit(extract->get_from_expr()));
  }

  RetType visitArrayOper(const hdk::ir::ArrayExpr* array_expr) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < array_expr->getElementCount(); ++i) {
      args_copy.push_back(visit(array_expr->getElement(i)));
    }
    const auto& type_info = array_expr->get_type_info();
    return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(
        type_info, args_copy, array_expr->isNull(), array_expr->isLocalAlloc());
  }

  RetType visitWindowFunction(const hdk::ir::WindowFunction* window_func) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (const auto& arg : window_func->getArgs()) {
      args_copy.push_back(visit(arg.get()));
    }
    std::vector<hdk::ir::ExprPtr> partition_keys_copy;
    for (const auto& partition_key : window_func->getPartitionKeys()) {
      partition_keys_copy.push_back(visit(partition_key.get()));
    }
    std::vector<hdk::ir::ExprPtr> order_keys_copy;
    for (const auto& order_key : window_func->getOrderKeys()) {
      order_keys_copy.push_back(visit(order_key.get()));
    }
    const auto& type_info = window_func->get_type_info();
    return hdk::ir::makeExpr<hdk::ir::WindowFunction>(type_info,
                                                      window_func->getKind(),
                                                      args_copy,
                                                      partition_keys_copy,
                                                      order_keys_copy,
                                                      window_func->getCollation());
  }

  RetType visitFunctionOper(const hdk::ir::FunctionOper* func_oper) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type_info = func_oper->get_type_info();
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(
        type_info, func_oper->getName(), args_copy);
  }

  RetType visitDatediffExpr(const hdk::ir::DatediffExpr* datediff) const override {
    return hdk::ir::makeExpr<hdk::ir::DatediffExpr>(datediff->get_type_info(),
                                                    datediff->get_field(),
                                                    visit(datediff->get_start_expr()),
                                                    visit(datediff->get_end_expr()));
  }

  RetType visitDateaddExpr(const hdk::ir::DateaddExpr* dateadd) const override {
    return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(dateadd->get_type_info(),
                                                   dateadd->get_field(),
                                                   visit(dateadd->get_number_expr()),
                                                   visit(dateadd->get_datetime_expr()));
  }

  RetType visitFunctionOperWithCustomTypeHandling(
      const hdk::ir::FunctionOperWithCustomTypeHandling* func_oper) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type_info = func_oper->get_type_info();
    return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
        type_info, func_oper->getName(), args_copy);
  }

  RetType visitLikelihood(const hdk::ir::LikelihoodExpr* likelihood) const override {
    return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(visit(likelihood->get_arg()),
                                                      likelihood->get_likelihood());
  }

  RetType visitAggExpr(const hdk::ir::AggExpr* agg) const override {
    RetType arg = agg->get_arg() ? visit(agg->get_arg()) : nullptr;
    return hdk::ir::makeExpr<hdk::ir::AggExpr>(agg->get_type_info(),
                                               agg->get_aggtype(),
                                               arg,
                                               agg->get_is_distinct(),
                                               agg->get_arg1());
  }

  RetType visitOffsetInFragment(const hdk::ir::OffsetInFragment*) const override {
    return hdk::ir::makeExpr<hdk::ir::OffsetInFragment>();
  }
};
