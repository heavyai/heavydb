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

class DeepCopyVisitor : public ScalarExprVisitor<std::shared_ptr<Analyzer::Expr>> {
 protected:
  typedef std::shared_ptr<Analyzer::Expr> RetType;
  RetType visitColumnVar(const Analyzer::ColumnVar* col_var) const override {
    return col_var->deep_copy();
  }

  RetType visitColumnVarTuple(
      const Analyzer::ExpressionTuple* col_var_tuple) const override {
    return col_var_tuple->deep_copy();
  }

  RetType visitVar(const Analyzer::Var* var) const override { return var->deep_copy(); }

  RetType visitConstant(const Analyzer::Constant* constant) const override {
    return constant->deep_copy();
  }

  RetType visitUOper(const Analyzer::UOper* uoper) const override {
    return makeExpr<Analyzer::UOper>(uoper->get_type_info(),
                                     uoper->get_contains_agg(),
                                     uoper->get_optype(),
                                     visit(uoper->get_operand()));
  }

  RetType visitBinOper(const Analyzer::BinOper* bin_oper) const override {
    return makeExpr<Analyzer::BinOper>(bin_oper->get_type_info(),
                                       bin_oper->get_contains_agg(),
                                       bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       visit(bin_oper->get_left_operand()),
                                       visit(bin_oper->get_right_operand()));
  }

  RetType visitInValues(const Analyzer::InValues* in_values) const override {
    const auto& value_list = in_values->get_value_list();
    std::list<RetType> new_list;
    for (const auto in_value : value_list) {
      new_list.push_back(visit(in_value.get()));
    }
    return makeExpr<Analyzer::InValues>(visit(in_values->get_arg()), new_list);
  }

  RetType visitInIntegerSet(const Analyzer::InIntegerSet* in_integer_set) const override {
    return makeExpr<Analyzer::InIntegerSet>(
        visit(in_integer_set->get_arg()),
        in_integer_set->get_value_list(),
        in_integer_set->get_type_info().get_notnull());
  }

  RetType visitCharLength(const Analyzer::CharLengthExpr* char_length) const override {
    return makeExpr<Analyzer::CharLengthExpr>(visit(char_length->get_arg()),
                                              char_length->get_calc_encoded_length());
  }

  RetType visitLikeExpr(const Analyzer::LikeExpr* like) const override {
    auto escape_expr = like->get_escape_expr();
    return makeExpr<Analyzer::LikeExpr>(visit(like->get_arg()),
                                        visit(like->get_like_expr()),
                                        escape_expr ? visit(escape_expr) : nullptr,
                                        like->get_is_ilike(),
                                        like->get_is_simple());
  }

  RetType visitRegexpExpr(const Analyzer::RegexpExpr* regexp) const override {
    auto escape_expr = regexp->get_escape_expr();
    return makeExpr<Analyzer::RegexpExpr>(visit(regexp->get_arg()),
                                          visit(regexp->get_pattern_expr()),
                                          escape_expr ? visit(escape_expr) : nullptr);
  }

  RetType visitCaseExpr(const Analyzer::CaseExpr* case_expr) const override {
    std::list<std::pair<RetType, RetType>> new_list;
    for (auto p : case_expr->get_expr_pair_list()) {
      new_list.push_back(std::make_pair(visit(p.first.get()), visit(p.second.get())));
    }
    auto else_expr = case_expr->get_else_expr();
    return makeExpr<Analyzer::CaseExpr>(
        case_expr->get_type_info(),
        case_expr->get_contains_agg(),
        new_list,
        else_expr == nullptr ? nullptr : visit(else_expr));
  }

  RetType visitDatetruncExpr(const Analyzer::DatetruncExpr* datetrunc) const override {
    return makeExpr<Analyzer::DatetruncExpr>(datetrunc->get_type_info(),
                                             datetrunc->get_contains_agg(),
                                             datetrunc->get_field(),
                                             visit(datetrunc->get_from_expr()));
  }

  RetType visitExtractExpr(const Analyzer::ExtractExpr* extract) const override {
    return makeExpr<Analyzer::ExtractExpr>(extract->get_type_info(),
                                           extract->get_contains_agg(),
                                           extract->get_field(),
                                           visit(extract->get_from_expr()));
  }

  RetType visitArrayOper(const Analyzer::ArrayExpr* array_expr) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
    for (size_t i = 0; i < array_expr->getElementCount(); ++i) {
      args_copy.push_back(visit(array_expr->getElement(i)));
    }
    const auto& type_info = array_expr->get_type_info();
    return makeExpr<Analyzer::ArrayExpr>(
        type_info, args_copy, array_expr->getExprIndex());
  }

  RetType visitFunctionOper(const Analyzer::FunctionOper* func_oper) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type_info = func_oper->get_type_info();
    return makeExpr<Analyzer::FunctionOper>(type_info, func_oper->getName(), args_copy);
  }

  RetType visitDatediffExpr(const Analyzer::DatediffExpr* datediff) const override {
    return makeExpr<Analyzer::DatediffExpr>(datediff->get_type_info(),
                                            datediff->get_field(),
                                            visit(datediff->get_start_expr()),
                                            visit(datediff->get_end_expr()));
  }

  RetType visitDateaddExpr(const Analyzer::DateaddExpr* dateadd) const override {
    return makeExpr<Analyzer::DateaddExpr>(dateadd->get_type_info(),
                                           dateadd->get_field(),
                                           visit(dateadd->get_number_expr()),
                                           visit(dateadd->get_datetime_expr()));
  }

  RetType visitFunctionOperWithCustomTypeHandling(
      const Analyzer::FunctionOperWithCustomTypeHandling* func_oper) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type_info = func_oper->get_type_info();
    return makeExpr<Analyzer::FunctionOperWithCustomTypeHandling>(
        type_info, func_oper->getName(), args_copy);
  }

  RetType visitLikelihood(const Analyzer::LikelihoodExpr* likelihood) const override {
    return makeExpr<Analyzer::LikelihoodExpr>(visit(likelihood->get_arg()),
                                              likelihood->get_likelihood());
  }

  RetType visitAggExpr(const Analyzer::AggExpr* agg) const override {
    RetType arg = agg->get_arg() ? visit(agg->get_arg()) : nullptr;
    return makeExpr<Analyzer::AggExpr>(agg->get_type_info(),
                                       agg->get_aggtype(),
                                       arg,
                                       agg->get_is_distinct(),
                                       agg->get_error_rate());
  }

  RetType visitOffsetInFragment(const Analyzer::OffsetInFragment*) const override {
    return makeExpr<Analyzer::OffsetInFragment>();
  }
};
