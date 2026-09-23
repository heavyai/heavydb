/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <QueryEngine/ScalarExprVisitor.h>

namespace Catalog_Namespace {
class Catalog;
}

namespace QueryRenderer {
/**
 * This is a utility class following a Visitor pattern that attempts to recursively build
 * out the SQL expression associated with a particular TargetEntry expr. Currently it's
 * main purpose is to rebuild the SQL expression from a render query during hit-testing.
 * For example, if the render query is so:
 *
 *   SELECT conv_4326_900913_x(lon) as x, conv_4326_900913_y(lat) as y, lang as color,
 * rowid FROM tweets_nov_feb WHERE ...
 *
 * And the user wanted to extract the 'x' column from that query during a hit-test, the
 * SQL expr 'conv_4326_900913_x(lon) as x' would need to be rebuilt for the hit-test
 * query:
 *
 *   SELECT conv_4326_900913_x(lon) as x FROM tweets_nov_feb WHERE rowid=<some rowid>
 *
 * This visitor class attempts to rebuild the sql expression for a particular output
 * column (or TargetEntry)
 *
 * This is currently a special use case for render hit-testing, but if useful could be
 * made into a more general utility class.
 */
class ProjHitTestColSqlBuilder : public ScalarExprVisitor<std::string> {
 public:
  ProjHitTestColSqlBuilder(const Analyzer::Expr& root, const std::string& col_name)
      : ScalarExprVisitor<std::string>(), root_(root), col_name_(col_name) {}

  // non-copyable
  ProjHitTestColSqlBuilder() = delete;
  ProjHitTestColSqlBuilder(const ProjHitTestColSqlBuilder&) = delete;
  const ProjHitTestColSqlBuilder& operator=(const ProjHitTestColSqlBuilder&) = delete;

  // builds the SQL expression string
  std::string build() const { return visit(&root_) + " as " + col_name_; }

 private:
  // root Expr to recurse
  const Analyzer::Expr& root_;

  // output name of the sql expression, will end with ".... AS <col_name_>"
  const std::string& col_name_;

  std::string visitVar(const Analyzer::Var* expr) const final;

  std::string visitColumnVar(const Analyzer::ColumnVar* expr) const final;

  std::string visitColumnVarTuple(const Analyzer::ExpressionTuple* expr) const final;

  std::string visitConstant(const Analyzer::Constant* expr) const final;

  std::string handleCast(const Analyzer::UOper& uoper,
                         const Analyzer::Expr& operand,
                         const std::string& operand_str) const;

  std::string visitUOper(const Analyzer::UOper* expr) const final;

  std::string visitBinOper(const Analyzer::BinOper* expr) const final;

  std::string visitInValues(const Analyzer::InValues* expr) const final;

  std::string visitInIntegerSet(const Analyzer::InIntegerSet* expr) const final;

  std::string visitCharLength(const Analyzer::CharLengthExpr* expr) const final;

  std::string visitKeyForString(
      const Analyzer::KeyForStringExpr* key_for_string) const final;

  std::string visitSampleRatio(const Analyzer::SampleRatioExpr* sample_ratio) const final;

  std::string visitStringOper(const Analyzer::StringOper* string_oper) const final;

  std::string visitCardinality(const Analyzer::CardinalityExpr* cardinality) const final;

  std::string visitDotProduct(const Analyzer::DotProductExpr* dot_product) const final;

  std::string visitLikeExpr(const Analyzer::LikeExpr* expr) const final;

  std::string visitRegexpExpr(const Analyzer::RegexpExpr* expr) const final;

  std::string visitCaseExpr(const Analyzer::CaseExpr* expr) const final;

  std::string visitDatetruncExpr(const Analyzer::DatetruncExpr* expr) const final;

  std::string visitExtractExpr(const Analyzer::ExtractExpr* expr) const final;

  std::string visitArrayOper(Analyzer::ArrayExpr const* expr) const final;

  std::string visitFunctionOper(const Analyzer::FunctionOper* expr) const final;

  std::string visitFunctionOperWithCustomTypeHandling(
      const Analyzer::FunctionOperWithCustomTypeHandling* expr) const final;

  std::string visitWindowFunction(
      const Analyzer::WindowFunction* window_func) const final;

  std::string visitGeoUOper(const Analyzer::GeoUOper* geo_expr) const final;

  std::string visitGeoBinOper(const Analyzer::GeoBinOper* geo_expr) const final;

  std::string visitGeoH3Oper(const Analyzer::GeoH3Oper* geo_expr) const final;

  std::string visitDatediffExpr(const Analyzer::DatediffExpr* expr) const final;

  std::string visitDateaddExpr(const Analyzer::DateaddExpr* expr) const final;

  std::string visitWidthBucket(const Analyzer::WidthBucketExpr* expr) const final;

  std::string visitLikelihood(const Analyzer::LikelihoodExpr* expr) const final;

  std::string visitOffsetInFragment(const Analyzer::OffsetInFragment* expr) const final;

  std::string visitFragmentId(const Analyzer::FragmentId* expr) const final;

  std::string visitFragmentIdAndOffset(
      const Analyzer::FragmentIdAndOffset* expr) const final;

  std::string visitAggExpr(const Analyzer::AggExpr* expr) const final;

  std::string visitGeoExpr(const Analyzer::GeoExpr* geo_expr) const final;

 protected:
  std::string aggregateResult(const std::string& aggregate,
                              const std::string& next_result) const final;
  std::string defaultResult() const final;

 private:
  std::string visitGeoConstant(const Analyzer::GeoConstant* geo_constant) const;
  std::string visitGeoOperator(const Analyzer::GeoOperator* geo_operator) const;
};

}  // namespace QueryRenderer
