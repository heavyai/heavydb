/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/HitTesting/ProjHitTestColSqlBuilder.h"
#include "Catalog/Catalog.h"

namespace QueryRenderer {

namespace {
inline std::string print_sql_type_info(const SQLTypeInfo& type_info) {
  return type_info.get_type_name() + "(" + std::to_string(type_info.get_precision()) +
         "," + std::to_string(type_info.get_scale()) + ") " +
         type_info.get_compression_name() + "(" +
         std::to_string(type_info.get_comp_param()) + ") ";
}

template <typename... Args>
inline bool function_name_match(const std::string& function_name,
                                Args&&... string_view_arg) {
  return ((function_name.compare(0, string_view_arg.size(), string_view_arg) == 0) ||
          ...);
};

}  // namespace

#define THROW_VISITOR_ERROR(expr, err_str)                                           \
  throw std::runtime_error(                                                          \
      (expr == &root_                                                                \
           ? "Error attempting to rebuild SQL projection expression for column \"" + \
                 col_name_ + "\". "                                                  \
           : "") +                                                                   \
      err_str);

#define THROW_UNSUPPORTED_VISITOR_ERROR(expr) \
  THROW_VISITOR_ERROR(expr, "Unsupported expr \"" + expr->toString() + "\"");

std::string ProjHitTestColSqlBuilder::visitColumnVar(
    const Analyzer::ColumnVar* expr) const {
  const auto& column_key = expr->getColumnKey();
  const auto cat = Catalog_Namespace::SysCatalog::instance().getCatalog(column_key.db_id);
  if (cat) {
    const auto td = cat->getMetadataForTable(column_key.table_id);
    if (td) {
      const auto cd =
          cat->getMetadataForColumn(column_key.table_id, column_key.column_id);
      if (cd) {
        return cat->name() + "." + td->tableName + "." + cd->columnName;
      }
      THROW_VISITOR_ERROR(expr,
                          "Invalid column id " + std::to_string(column_key.column_id) +
                              " in expr \"" + expr->toString() + "\".");
    }
    THROW_VISITOR_ERROR(expr,
                        "Invalid table id " + std::to_string(column_key.table_id) +
                            " in expr \"" + expr->toString() + "\".");
  }
  THROW_VISITOR_ERROR(expr,
                      "Invalid database id " + std::to_string(column_key.db_id) +
                          " in expr \"" + expr->toString() + "\".");
}

std::string ProjHitTestColSqlBuilder::visitConstant(
    const Analyzer::Constant* expr) const {
  if (expr->get_is_null()) {
    return "NULL";
  }
  // TODO(croot): handle array constants?
  std::string datum_str;
  const auto& datum_type = expr->get_type_info();
  try {
    datum_str = DatumToString(expr->get_constval(), datum_type);
  } catch (std::exception& err) {
    THROW_VISITOR_ERROR(
        expr, "Unsupported constant expr \"" + expr->toString() + "\". " + err.what());
  }
  if (IS_STRING(datum_type.get_type()) && (!datum_str.size() || datum_str[0] != '\'')) {
    return "'" + datum_str + "'";
  }
  return datum_str;
}

std::string ProjHitTestColSqlBuilder::handleCast(const Analyzer::UOper& uoper,
                                                 const Analyzer::Expr& operand,
                                                 const std::string& operand_str) const {
  const auto& cast_type = uoper.get_type_info();
  const auto& operand_type = operand.get_type_info();
  if ((operand_type.get_type() == cast_type.get_type()) ||
      (operand_type.is_string() && cast_type.is_string())) {
    // no explicit cast needed since types are the same. It's just an internal cast, so
    // pass the operand along. Any errors should be caught by calcite.
    return operand_str;
  } else if (cast_type.is_number()) {
    return "CAST(" + operand_str + " AS " + cast_type.get_type_name() + ")";
  } else {
    // Note that other scalar types, like time, are not supported in vega scales
    // currently. When those types are supported, this function will need to be updated to
    // handle timestamp casts. Because `get_type_name()` includes timestamp precision, we
    // should be able to use the branch above.
    THROW_VISITOR_ERROR(&uoper,
                        "Unsupported CAST of type: " + print_sql_type_info(cast_type) +
                            " on operand " + operand.toString());
  }
}

std::string ProjHitTestColSqlBuilder::visitUOper(const Analyzer::UOper* expr) const {
  auto operand_str = visit(expr->get_operand());
  switch (expr->get_optype()) {
    case kNOT:
      return "NOT (" + operand_str + ")";
    case kUMINUS:
      return "-(" + operand_str + ")";
    case kISNULL:
      return "(" + operand_str + ") IS NULL";
    case kISNOTNULL:
      return "(" + operand_str + ") IS NOT NULL";
    case kCAST:
      return handleCast(*expr, *expr->get_operand(), operand_str);
    default:
      THROW_VISITOR_ERROR(expr,
                          "Unsupported unary operator " +
                              std::to_string(expr->get_optype()) + ". " +
                              expr->toString());
  }
  CHECK(false);
  return "";
}

std::string ProjHitTestColSqlBuilder::visitBinOper(const Analyzer::BinOper* expr) const {
  std::string op;
  switch (expr->get_optype()) {
    case kEQ:
      op = "= ";
      break;
    case kNE:
      op = "<> ";
      break;
    case kLT:
      op = "< ";
      break;
    case kLE:
      op = "<= ";
      break;
    case kGT:
      op = "> ";
      break;
    case kGE:
      op = ">= ";
      break;
    case kAND:
      op = "AND ";
      break;
    case kOR:
      op = "OR ";
      break;
    case kMINUS:
      op = "- ";
      break;
    case kPLUS:
      op = "+ ";
      break;
    case kMULTIPLY:
      op = "* ";
      break;
    case kDIVIDE:
      op = "/ ";
      break;
    case kMODULO:
      op = "% ";
      break;
    case kARRAY_AT:
      op = "[] ";
      break;
    default:
      break;
  }
  std::string rtn;
  try {
    rtn = "(" + visit(expr->get_left_operand());
  } catch (std::exception& err) {
    THROW_VISITOR_ERROR(
        expr,
        "Invalid left operand in \"" + expr->toString() + "\". Error: " + err.what());
  }
  rtn += " " + op;

  switch (expr->get_qualifier()) {
    case kANY:
      rtn += "ANY ";
      break;
    case kALL:
      rtn += "ALL ";
      break;
    case kONE:
      break;
  }
  try {
    rtn += visit(expr->get_right_operand()) + ")";
  } catch (std::exception& err) {
    THROW_VISITOR_ERROR(
        expr,
        "Invalid right operand in \"" + expr->toString() + "\". Error: " + err.what());
  }
  return rtn;
}

std::string ProjHitTestColSqlBuilder::visitCaseExpr(
    const Analyzer::CaseExpr* expr) const {
  const auto& expr_pair_list = expr->get_expr_pair_list();
  if (expr_pair_list.size()) {
    std::string rtn = "CASE";
    try {
      for (const auto& expr_pair : expr_pair_list) {
        rtn += " WHEN " + visit(expr_pair.first.get()) + " THEN " +
               visit(expr_pair.second.get());
      }
      auto else_expr = expr->get_else_expr();
      if (else_expr) {
        rtn += " ELSE " + visit(else_expr);
      }
    } catch (std::exception& err) {
      THROW_VISITOR_ERROR(
          expr, "Invalid sub-expr in \"" + expr->toString() + "\". Error: " + err.what());
    }
    rtn += " END";
    return rtn;
  }
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitStringOper(
    const Analyzer::StringOper* string_oper) const {
  const auto function_name = toString(string_oper->get_kind());
  std::string rtn = function_name + "(";
  const auto arity = string_oper->getArity();
  if (arity) {
    size_t i = 0;
    try {
      rtn += visit(string_oper->getArg(i++));
      for (; i < arity; ++i) {
        rtn += ", " + visit(string_oper->getArg(i));
      }
    } catch (std::exception& err) {
      THROW_VISITOR_ERROR(string_oper,
                          "Invalid argument " + std::to_string(i) + " in \"" +
                              string_oper->toString() + "\". Error: " + err.what());
    }
  }
  rtn += ")";
  return rtn;
}

std::string ProjHitTestColSqlBuilder::visitFunctionOper(
    const Analyzer::FunctionOper* expr) const {
  using namespace std::literals;

  auto function_name = expr->getName();
  auto arity = expr->getArity();
  if (function_name.compare(0, 3, "ST_") == 0) {
    // geo ST_* functions may need special handling, see below.

    if (function_name_match(function_name, "ST_X_Point"sv, "ST_Y_Point"sv)) {
      // ST_X_Point/ST_Y_Point are specialized internal functions used after translating
      // ST_X/ST_Y See: RelAlgTranslator::translateFunction() This is a hacky stop-gap to
      // fix https://omnisci.atlassian.net/browse/BE-5687 This fixes ST_X/ST_Y
      // specifically, but there is a fundamental problem using Analyzer::Expr nodes to
      // rebuild sql expressions as there isn't a direct correlation due to the internal
      // translation operators. Ultimately we need to find a better way to build hit-test
      // queries. This most likely means replacing Analyzer::Expr nodes with an RA nodes
      // representation since RA has a direct correlation to sql. RA also gives us the
      // opportunity to skip building sql altogether and instead just build RA directly,
      // bypassing calcite parsing entirely.

      // brute force convert from ST_X_Point -> ST_X (similarly for Y)
      function_name.erase(4, function_name.size());

      // it is known that the first argument to ST_X_Point is the geo point column name.
      // That's the only arg to forward to ST_X, so limit the arg forwarding to just the
      // first 1 (similarly for Y)
      CHECK_GE(arity, 1u);
      arity = 1;
    } else if (function_name_match(function_name,
                                   "ST_XMin"sv,
                                   "ST_YMin"sv,
                                   "ST_XMax"sv,
                                   "ST_YMax"sv,
                                   "ST_NRings"sv,
                                   "ST_NumGeometries"sv,
                                   "ST_NPoints"sv,
                                   "ST_Length"sv,
                                   "ST_Perimeter"sv,
                                   "ST_Area"sv,
                                   "ST_Distance"sv,
                                   "ST_MaxDistance"sv,
                                   "ST_Intersects"sv,
                                   "ST_Disjoint"sv,
                                   "ST_Contains"sv,
                                   "ST_IntersectsBox"sv,
                                   "ST_Within"sv,
                                   "ST_DWithin"sv,
                                   "ST_DFullyWithin"sv,
                                   "ST_GeomFromText"sv,
                                   "ST_GeogFromText"sv,
                                   "ST_Point"sv,
                                   "ST_Centroid"sv,
                                   "ST_SetSRID"sv,
                                   "ST_Intersection"sv,
                                   "ST_Difference"sv,
                                   "ST_Union"sv,
                                   "ST_Buffer"sv,
                                   "ST_IsEmpty"sv,
                                   "ST_IsValid"sv)) {
      // Throw an error for a handful of known translated geo functions. These can be
      // found in RelAlgTranslator::translateFunction
      THROW_VISITOR_ERROR(expr,
                          "Geo function " + function_name + " in \"" + expr->toString() +
                              "\" is not currently supported for hit-testing.");
    }
  }
  std::string rtn = function_name + "(";
  if (arity) {
    size_t i = 0;
    try {
      rtn += visit(expr->getArg(i++));
      for (; i < arity; ++i) {
        rtn += ", " + visit(expr->getArg(i));
      }
    } catch (std::exception& err) {
      THROW_VISITOR_ERROR(expr,
                          "Invalid argument " + std::to_string(i) + " in \"" +
                              expr->toString() + "\". Error: " + err.what());
    }
  }
  rtn += ")";
  return rtn;
}

std::string ProjHitTestColSqlBuilder::visitVar(const Analyzer::Var* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitDatetruncExpr(
    const Analyzer::DatetruncExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitExtractExpr(
    const Analyzer::ExtractExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitArrayOper(
    Analyzer::ArrayExpr const* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitColumnVarTuple(
    const Analyzer::ExpressionTuple* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitInValues(
    const Analyzer::InValues* expr) const {
  std::string rtn;
  try {
    rtn = visit(expr->get_arg());
  } catch (std::exception& err) {
    THROW_VISITOR_ERROR(
        expr, "Invalid arg in \"" + expr->toString() + "\". Error: " + err.what());
  }
  const auto& values = expr->get_value_list();
  if (values.size()) {
    rtn += " IN (";
    try {
      auto itr = values.begin();
      rtn += visit(itr->get());
      for (++itr; itr != values.end(); ++itr) {
        rtn += ", " + visit(itr->get());
      }
    } catch (std::exception& err) {
      THROW_VISITOR_ERROR(
          expr, "Invalid value in \"" + expr->toString() + "\". Error: " + err.what());
    }
    rtn += ")";
  }
  return rtn;
}

std::string ProjHitTestColSqlBuilder::visitInIntegerSet(
    const Analyzer::InIntegerSet* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitCharLength(
    const Analyzer::CharLengthExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitKeyForString(
    const Analyzer::KeyForStringExpr* key_for_string) const {
  return "key_for_string(" + visit(key_for_string->get_arg()) + ")";
}

std::string ProjHitTestColSqlBuilder::visitSampleRatio(
    const Analyzer::SampleRatioExpr* sample_ratio) const {
  return "sample_ratio(" + visit(sample_ratio->get_arg()) + ")";
}

std::string ProjHitTestColSqlBuilder::visitCardinality(
    const Analyzer::CardinalityExpr* cardinality) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(cardinality);
}

std::string ProjHitTestColSqlBuilder::visitDotProduct(
    const Analyzer::DotProductExpr* dot_product) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(dot_product);
}

std::string ProjHitTestColSqlBuilder::visitLikeExpr(
    const Analyzer::LikeExpr* expr) const {
  std::string like_expr_str =
      visit(expr->get_arg()) + " " + (expr->get_is_ilike() ? "ILIKE" : "LIKE");
  std::string like_str = visit(expr->get_like_expr());
  if (expr->get_is_simple()) {
    if (like_str.size() && like_str[0] == '\'' && like_str[like_str.size() - 1] == '\'') {
      // A 'simple' like expression means we can use fast path search (fits '%str%'
      // pattern with no inner '%','_','[',']' - see:
      // https://github.com/omnisci/omniscidb-internal/blob/v5.1.2/Analyzer/Analyzer.h#L908
      // If the like expession is a string literal, the returned string from the above
      // visit call will be a single-quoted string. We'll need to inject the '%' just
      // inside the single quotes.
      like_str.insert(1, "%");
      like_str.insert(like_str.size() - 1, "%");
    }
  }
  like_expr_str += " " + like_str;
  auto esc_expr = expr->get_escape_expr();
  if (esc_expr) {
    like_expr_str += " ESCAPE " + visit(esc_expr);
  }
  return like_expr_str;
}

std::string ProjHitTestColSqlBuilder::visitRegexpExpr(
    const Analyzer::RegexpExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitFunctionOperWithCustomTypeHandling(
    const Analyzer::FunctionOperWithCustomTypeHandling* expr) const {
  return visitFunctionOper(expr);
}

std::string ProjHitTestColSqlBuilder::visitWindowFunction(
    const Analyzer::WindowFunction* window_func) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(window_func);
}

std::string ProjHitTestColSqlBuilder::visitGeoUOper(
    const Analyzer::GeoUOper* geo_expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(geo_expr);
}

std::string ProjHitTestColSqlBuilder::visitGeoBinOper(
    const Analyzer::GeoBinOper* geo_expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(geo_expr);
}

std::string ProjHitTestColSqlBuilder::visitGeoH3Oper(
    const Analyzer::GeoH3Oper* geo_expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(geo_expr);
}

std::string ProjHitTestColSqlBuilder::visitDatediffExpr(
    const Analyzer::DatediffExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitDateaddExpr(
    const Analyzer::DateaddExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitWidthBucket(
    const Analyzer::WidthBucketExpr* expr) const {
  return "width_bucket(" + visit(expr->get_target_value()) + ", " +
         visit(expr->get_lower_bound()) + ", " + visit(expr->get_upper_bound()) + ", " +
         visit(expr->get_partition_count()) + ")";
}

std::string ProjHitTestColSqlBuilder::visitLikelihood(
    const Analyzer::LikelihoodExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitOffsetInFragment(
    const Analyzer::OffsetInFragment* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitFragmentId(
    const Analyzer::FragmentId* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitFragmentIdAndOffset(
    const Analyzer::FragmentIdAndOffset* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitAggExpr(const Analyzer::AggExpr* expr) const {
  THROW_UNSUPPORTED_VISITOR_ERROR(expr);
}

std::string ProjHitTestColSqlBuilder::visitGeoExpr(
    const Analyzer::GeoExpr* geo_expr) const {
  const auto geo_operator = dynamic_cast<const Analyzer::GeoOperator*>(geo_expr);
  if (geo_operator) {
    return visitGeoOperator(geo_operator);
  }

  const auto geo_constant = dynamic_cast<const Analyzer::GeoConstant*>(geo_expr);
  if (geo_constant) {
    return visitGeoConstant(geo_constant);
  }

  THROW_UNSUPPORTED_VISITOR_ERROR(geo_expr);
}

std::string ProjHitTestColSqlBuilder::visitGeoConstant(
    const Analyzer::GeoConstant* geo_constant) const {
  return "ST_GeomFromText('" + geo_constant->getWKTString() + "', " +
         std::to_string(geo_constant->get_type_info().get_input_srid()) + ")";
}

std::string ProjHitTestColSqlBuilder::visitGeoOperator(
    const Analyzer::GeoOperator* geo_operator) const {
  auto const arity = geo_operator->size();
  std::string rtn = geo_operator->getName() + "(";
  if (arity) {
    size_t i = 0;
    try {
      rtn += visit(geo_operator->getOperand(i++));
      for (; i < arity; ++i) {
        rtn += ", " + visit(geo_operator->getOperand(i));
      }
    } catch (std::exception& err) {
      THROW_VISITOR_ERROR(geo_operator,
                          "Invalid argument " + std::to_string(i) + " in \"" +
                              geo_operator->toString() + "\". Error: " + err.what());
    }
  }
  rtn += ")";
  return rtn;
}

std::string ProjHitTestColSqlBuilder::aggregateResult(
    const std::string& aggregate,
    const std::string& next_result) const {
  throw std::runtime_error(
      "Error attempting to rebuild SQL projection expression for column \"" + col_name_ +
      "\". aggregateResult() should not be called. Root expr tree: " + root_.toString());
}

std::string ProjHitTestColSqlBuilder::defaultResult() const {
  throw std::runtime_error(
      "Error attempting to rebuild SQL projection expression for column \"" + col_name_ +
      "\". defaultResult() should not be called. Root expr tree: " + root_.toString());
}

}  // namespace QueryRenderer
