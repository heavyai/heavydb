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

#ifndef QUERYENGINE_SCALAREXPRVISITOR_H
#define QUERYENGINE_SCALAREXPRVISITOR_H

#include "../Analyzer/Analyzer.h"

template <class T>
class ScalarExprVisitor {
 public:
  T visit(const Analyzer::Expr* expr) const {
    CHECK(expr);
    visitBegin();
    const auto var = dynamic_cast<const Analyzer::Var*>(expr);
    if (var) {
      return visitVar(var);
    }
    const auto column_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
    if (column_var) {
      return visitColumnVar(column_var);
    }
    const auto column_var_tuple = dynamic_cast<const Analyzer::ExpressionTuple*>(expr);
    if (column_var_tuple) {
      return visitColumnVarTuple(column_var_tuple);
    }
    const auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
    if (constant) {
      return visitConstant(constant);
    }
    const auto uoper = dynamic_cast<const Analyzer::UOper*>(expr);
    if (uoper) {
      return visitUOper(uoper);
    }
    const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
    if (bin_oper) {
      return visitBinOper(bin_oper);
    }
    const auto geo_expr = dynamic_cast<const Analyzer::GeoExpr*>(expr);
    if (geo_expr) {
      return visitGeoExpr(geo_expr);
    }
    const auto in_values = dynamic_cast<const Analyzer::InValues*>(expr);
    if (in_values) {
      return visitInValues(in_values);
    }
    const auto in_integer_set = dynamic_cast<const Analyzer::InIntegerSet*>(expr);
    if (in_integer_set) {
      return visitInIntegerSet(in_integer_set);
    }
    const auto char_length = dynamic_cast<const Analyzer::CharLengthExpr*>(expr);
    if (char_length) {
      return visitCharLength(char_length);
    }
    const auto key_for_string = dynamic_cast<const Analyzer::KeyForStringExpr*>(expr);
    if (key_for_string) {
      return visitKeyForString(key_for_string);
    }
    const auto sample_ratio = dynamic_cast<const Analyzer::SampleRatioExpr*>(expr);
    if (sample_ratio) {
      return visitSampleRatio(sample_ratio);
    }
    const auto width_bucket = dynamic_cast<const Analyzer::WidthBucketExpr*>(expr);
    if (width_bucket) {
      return visitWidthBucket(width_bucket);
    }
    const auto string_oper = dynamic_cast<const Analyzer::StringOper*>(expr);
    if (string_oper) {
      return visitStringOper(string_oper);
    }
    const auto cardinality = dynamic_cast<const Analyzer::CardinalityExpr*>(expr);
    if (cardinality) {
      return visitCardinality(cardinality);
    }
    const auto width_bucket_expr = dynamic_cast<const Analyzer::WidthBucketExpr*>(expr);
    if (width_bucket_expr) {
      return visitWidthBucket(width_bucket_expr);
    }
    const auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
    if (like_expr) {
      return visitLikeExpr(like_expr);
    }
    const auto regexp_expr = dynamic_cast<const Analyzer::RegexpExpr*>(expr);
    if (regexp_expr) {
      return visitRegexpExpr(regexp_expr);
    }
    const auto case_ = dynamic_cast<const Analyzer::CaseExpr*>(expr);
    if (case_) {
      return visitCaseExpr(case_);
    }
    const auto datetrunc = dynamic_cast<const Analyzer::DatetruncExpr*>(expr);
    if (datetrunc) {
      return visitDatetruncExpr(datetrunc);
    }
    const auto extract = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
    if (extract) {
      return visitExtractExpr(extract);
    }
    const auto window_func = dynamic_cast<const Analyzer::WindowFunction*>(expr);
    if (window_func) {
      return visitWindowFunction(window_func);
    }
    const auto func_with_custom_type_handling =
        dynamic_cast<const Analyzer::FunctionOperWithCustomTypeHandling*>(expr);
    if (func_with_custom_type_handling) {
      return visitFunctionOperWithCustomTypeHandling(func_with_custom_type_handling);
    }
    const auto func = dynamic_cast<const Analyzer::FunctionOper*>(expr);
    if (func) {
      return visitFunctionOper(func);
    }
    const auto array = dynamic_cast<const Analyzer::ArrayExpr*>(expr);
    if (array) {
      return visitArrayOper(array);
    }
    const auto geo_uop = dynamic_cast<const Analyzer::GeoUOper*>(expr);
    if (geo_uop) {
      return visitGeoUOper(geo_uop);
    }
    const auto geo_binop = dynamic_cast<const Analyzer::GeoBinOper*>(expr);
    if (geo_binop) {
      return visitGeoBinOper(geo_binop);
    }
    const auto datediff = dynamic_cast<const Analyzer::DatediffExpr*>(expr);
    if (datediff) {
      return visitDatediffExpr(datediff);
    }
    const auto dateadd = dynamic_cast<const Analyzer::DateaddExpr*>(expr);
    if (dateadd) {
      return visitDateaddExpr(dateadd);
    }
    const auto likelihood = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
    if (likelihood) {
      return visitLikelihood(likelihood);
    }
    const auto offset_in_fragment = dynamic_cast<const Analyzer::OffsetInFragment*>(expr);
    if (offset_in_fragment) {
      return visitOffsetInFragment(offset_in_fragment);
    }
    const auto agg = dynamic_cast<const Analyzer::AggExpr*>(expr);
    if (agg) {
      return visitAggExpr(agg);
    }
    const auto range_join_oper = dynamic_cast<const Analyzer::RangeOper*>(expr);
    if (range_join_oper) {
      return visitRangeJoinOper(range_join_oper);
    }
    return defaultResult();
  }

 protected:
  virtual T visitVar(const Analyzer::Var*) const { return defaultResult(); }

  virtual T visitColumnVar(const Analyzer::ColumnVar*) const { return defaultResult(); }

  virtual T visitColumnVarTuple(const Analyzer::ExpressionTuple*) const {
    return defaultResult();
  }

  virtual T visitConstant(const Analyzer::Constant*) const { return defaultResult(); }

  virtual T visitUOper(const Analyzer::UOper* uoper) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(uoper->get_operand()));
    return result;
  }

  virtual T visitBinOper(const Analyzer::BinOper* bin_oper) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(bin_oper->get_left_operand()));
    result = aggregateResult(result, visit(bin_oper->get_right_operand()));
    return result;
  }

  virtual T visitGeoExpr(const Analyzer::GeoExpr* geo_expr) const {
    T result = defaultResult();
    const auto geo_expr_children = geo_expr->getChildExprs();
    for (const auto expr : geo_expr_children) {
      result = aggregateResult(result, visit(expr));
    }
    return result;
  }

  virtual T visitInValues(const Analyzer::InValues* in_values) const {
    T result = visit(in_values->get_arg());
    const auto& value_list = in_values->get_value_list();
    for (const auto& in_value : value_list) {
      result = aggregateResult(result, visit(in_value.get()));
    }
    return result;
  }

  virtual T visitInIntegerSet(const Analyzer::InIntegerSet* in_integer_set) const {
    return visit(in_integer_set->get_arg());
  }

  virtual T visitCharLength(const Analyzer::CharLengthExpr* char_length) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(char_length->get_arg()));
    return result;
  }

  virtual T visitKeyForString(const Analyzer::KeyForStringExpr* key_for_string) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(key_for_string->get_arg()));
    return result;
  }

  virtual T visitSampleRatio(const Analyzer::SampleRatioExpr* sample_ratio) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(sample_ratio->get_arg()));
    return result;
  }

  virtual T visitStringOper(const Analyzer::StringOper* string_oper) const {
    T result = defaultResult();
    for (const auto& arg : string_oper->getOwnArgs()) {
      result = aggregateResult(result, visit(arg.get()));
    }
    return result;
  }

  virtual T visitCardinality(const Analyzer::CardinalityExpr* cardinality) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(cardinality->get_arg()));
    return result;
  }

  virtual T visitLikeExpr(const Analyzer::LikeExpr* like) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(like->get_arg()));
    result = aggregateResult(result, visit(like->get_like_expr()));
    if (like->get_escape_expr()) {
      result = aggregateResult(result, visit(like->get_escape_expr()));
    }
    return result;
  }

  virtual T visitRegexpExpr(const Analyzer::RegexpExpr* regexp) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(regexp->get_arg()));
    result = aggregateResult(result, visit(regexp->get_pattern_expr()));
    if (regexp->get_escape_expr()) {
      result = aggregateResult(result, visit(regexp->get_escape_expr()));
    }
    return result;
  }

  virtual T visitWidthBucket(const Analyzer::WidthBucketExpr* width_bucket_expr) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(width_bucket_expr->get_target_value()));
    result = aggregateResult(result, visit(width_bucket_expr->get_lower_bound()));
    result = aggregateResult(result, visit(width_bucket_expr->get_upper_bound()));
    result = aggregateResult(result, visit(width_bucket_expr->get_partition_count()));
    return result;
  }

  virtual T visitCaseExpr(const Analyzer::CaseExpr* case_) const {
    T result = defaultResult();
    const auto& expr_pair_list = case_->get_expr_pair_list();
    for (const auto& expr_pair : expr_pair_list) {
      result = aggregateResult(result, visit(expr_pair.first.get()));
      result = aggregateResult(result, visit(expr_pair.second.get()));
    }
    result = aggregateResult(result, visit(case_->get_else_expr()));
    return result;
  }

  virtual T visitDatetruncExpr(const Analyzer::DatetruncExpr* datetrunc) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(datetrunc->get_from_expr()));
    return result;
  }

  virtual T visitExtractExpr(const Analyzer::ExtractExpr* extract) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(extract->get_from_expr()));
    return result;
  }

  virtual T visitFunctionOperWithCustomTypeHandling(
      const Analyzer::FunctionOperWithCustomTypeHandling* func_oper) const {
    return visitFunctionOper(func_oper);
  }

  virtual T visitArrayOper(Analyzer::ArrayExpr const* array_expr) const {
    T result = defaultResult();
    for (size_t i = 0; i < array_expr->getElementCount(); ++i) {
      result = aggregateResult(result, visit(array_expr->getElement(i)));
    }
    return result;
  }

  virtual T visitGeoUOper(const Analyzer::GeoUOper* geo_expr) const {
    T result = defaultResult();
    for (const auto& arg : geo_expr->getArgs0()) {
      result = aggregateResult(result, visit(arg.get()));
    }
    return result;
  }

  virtual T visitGeoBinOper(const Analyzer::GeoBinOper* geo_expr) const {
    T result = defaultResult();
    for (const auto& arg : geo_expr->getArgs0()) {
      result = aggregateResult(result, visit(arg.get()));
    }
    for (const auto& arg : geo_expr->getArgs1()) {
      result = aggregateResult(result, visit(arg.get()));
    }
    return result;
  }

  virtual T visitFunctionOper(const Analyzer::FunctionOper* func_oper) const {
    T result = defaultResult();
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      result = aggregateResult(result, visit(func_oper->getArg(i)));
    }
    return result;
  }

  virtual T visitWindowFunction(const Analyzer::WindowFunction* window_func) const {
    T result = defaultResult();
    for (const auto& arg : window_func->getArgs()) {
      result = aggregateResult(result, visit(arg.get()));
    }
    for (const auto& partition_key : window_func->getPartitionKeys()) {
      result = aggregateResult(result, visit(partition_key.get()));
    }
    for (const auto& order_key : window_func->getOrderKeys()) {
      result = aggregateResult(result, visit(order_key.get()));
    }
    return result;
  }

  virtual T visitDatediffExpr(const Analyzer::DatediffExpr* datediff) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(datediff->get_start_expr()));
    result = aggregateResult(result, visit(datediff->get_end_expr()));
    return result;
  }

  virtual T visitDateaddExpr(const Analyzer::DateaddExpr* dateadd) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(dateadd->get_number_expr()));
    result = aggregateResult(result, visit(dateadd->get_datetime_expr()));
    return result;
  }

  virtual T visitLikelihood(const Analyzer::LikelihoodExpr* likelihood) const {
    return visit(likelihood->get_arg());
  }

  virtual T visitOffsetInFragment(const Analyzer::OffsetInFragment*) const {
    return defaultResult();
  }

  virtual T visitAggExpr(const Analyzer::AggExpr* agg) const {
    T result = defaultResult();
    if (agg->get_arg()) {
      return aggregateResult(result, visit(agg->get_arg()));
    }
    return defaultResult();
  }

  virtual T visitRangeJoinOper(const Analyzer::RangeOper* range_oper) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(range_oper->get_left_operand()));
    result = aggregateResult(result, visit(range_oper->get_right_operand()));
    return result;
  }

 protected:
  virtual T aggregateResult(const T& aggregate, const T& next_result) const {
    return next_result;
  }

  virtual void visitBegin() const {}

  virtual T defaultResult() const { return T{}; }
};

#endif  // QUERYENGINE_SCALAREXPRVISITOR_H
