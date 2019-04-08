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

#include "CalciteAdapter.h"
#include "CalciteDeserializerUtils.h"

#include "../Parser/ParserNode.h"
#include "../Shared/StringTransform.h"

#include <rapidjson/document.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>

#include <set>
#include <unordered_map>
#include <unordered_set>

namespace {

ssize_t get_agg_operand_idx(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  CHECK(expr.HasMember("agg"));
  const auto& agg_operands = expr["operands"];
  CHECK(agg_operands.IsArray());
  CHECK(agg_operands.Size() <= 2);
  return agg_operands.Empty() ? -1 : agg_operands[0].GetInt();
}

std::tuple<const rapidjson::Value*, SQLTypeInfo, SQLTypeInfo> parse_literal(
    const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  auto val_it = expr.FindMember("literal");
  CHECK(val_it != expr.MemberEnd());
  auto type_it = expr.FindMember("type");
  CHECK(type_it != expr.MemberEnd());
  CHECK(type_it->value.IsString());
  const auto type_name = std::string(type_it->value.GetString());
  auto target_type_it = expr.FindMember("target_type");
  CHECK(target_type_it != expr.MemberEnd());
  CHECK(target_type_it->value.IsString());
  const auto target_type_name = std::string(target_type_it->value.GetString());
  auto scale_it = expr.FindMember("scale");
  CHECK(scale_it != expr.MemberEnd());
  CHECK(scale_it->value.IsInt());
  const int scale = scale_it->value.GetInt();
  auto type_scale_it = expr.FindMember("type_scale");
  CHECK(type_scale_it != expr.MemberEnd());
  CHECK(type_scale_it->value.IsInt());
  const int type_scale = type_scale_it->value.GetInt();
  auto precision_it = expr.FindMember("precision");
  CHECK(precision_it != expr.MemberEnd());
  CHECK(precision_it->value.IsInt());
  const int precision = precision_it->value.GetInt();
  auto type_precision_it = expr.FindMember("type_precision");
  CHECK(type_precision_it != expr.MemberEnd());
  CHECK(type_precision_it->value.IsInt());
  const int type_precision = type_precision_it->value.GetInt();
  SQLTypeInfo ti(to_sql_type(type_name), precision, scale, false);
  SQLTypeInfo target_ti(to_sql_type(target_type_name), type_precision, type_scale, false);
  return std::make_tuple(&(val_it->value), ti, target_ti);
}

std::shared_ptr<Analyzer::Expr> set_transient_dict(
    const std::shared_ptr<Analyzer::Expr> expr) {
  const auto& ti = expr->get_type_info();
  if (!ti.is_string() || ti.get_compression() != kENCODING_NONE) {
    return expr;
  }
  auto transient_dict_ti = ti;
  transient_dict_ti.set_compression(kENCODING_DICT);
  transient_dict_ti.set_comp_param(TRANSIENT_DICT_ID);
  transient_dict_ti.set_fixed_size();
  return expr->add_cast(transient_dict_ti);
}

class CalciteAdapter {
 public:
  CalciteAdapter(const Catalog_Namespace::Catalog& cat, const rapidjson::Value& rels)
      : cat_(cat) {
    time(&now_);
    CHECK(rels.IsArray());
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& scan_ra = *rels_it;
      CHECK(scan_ra.IsObject());
      if (scan_ra["relOp"].GetString() != std::string("EnumerableTableScan")) {
        break;
      }
      col_names_.emplace_back(
          ColNames{getColNames(scan_ra), getTableFromScanNode(scan_ra)});
    }
  }

  CalciteAdapter(const CalciteAdapter&) = delete;

  CalciteAdapter& operator=(const CalciteAdapter&) = delete;

  std::shared_ptr<Analyzer::Expr> getExprFromNode(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    if (expr.IsObject() && expr.HasMember("op")) {
      return translateOp(expr, scan_targets);
    }
    if (expr.IsObject() && expr.HasMember("input")) {
      return translateColRef(expr, scan_targets);
    }
    if (expr.IsObject() && expr.HasMember("agg")) {
      return translateAggregate(expr, scan_targets);
    }
    if (expr.IsObject() && expr.HasMember("literal")) {
      return translateTypedLiteral(expr);
    }
    throw std::runtime_error("Unsupported node type");
  }

  std::pair<std::shared_ptr<Analyzer::Expr>, SQLQualifier> getQuantifiedRhs(
      const rapidjson::Value& rhs_op,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    std::shared_ptr<Analyzer::Expr> rhs;
    SQLQualifier sql_qual{kONE};
    const auto rhs_op_it = rhs_op.FindMember("op");
    if (rhs_op_it == rhs_op.MemberEnd()) {
      return std::make_pair(rhs, sql_qual);
    }
    CHECK(rhs_op_it->value.IsString());
    const auto& qual_str = rhs_op_it->value.GetString();
    const auto rhs_op_operands_it = rhs_op.FindMember("operands");
    CHECK(rhs_op_operands_it != rhs_op.MemberEnd());
    const auto& rhs_op_operands = rhs_op_operands_it->value;
    CHECK(rhs_op_operands.IsArray());
    if (qual_str == std::string("PG_ANY") || qual_str == std::string("PG_ALL")) {
      CHECK_EQ(unsigned(1), rhs_op_operands.Size());
      rhs = getExprFromNode(rhs_op_operands[0], scan_targets);
      sql_qual = qual_str == std::string("PG_ANY") ? kANY : kALL;
    }
    if (!rhs && qual_str == std::string("CAST")) {
      CHECK_EQ(unsigned(1), rhs_op_operands.Size());
      std::tie(rhs, sql_qual) = getQuantifiedRhs(rhs_op_operands[0], scan_targets);
    }
    return std::make_pair(rhs, sql_qual);
  }

  std::shared_ptr<Analyzer::Expr> translateOp(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto op_str = expr["op"].GetString();
    if (op_str == std::string("LIKE") || op_str == std::string("PG_ILIKE")) {
      return translateLike(expr, scan_targets, op_str == std::string("PG_ILIKE"));
    }
    if (op_str == std::string("REGEXP_LIKE")) {
      return translateRegexp(expr, scan_targets);
    }
    if (op_str == std::string("LIKELY")) {
      return translateLikely(expr, scan_targets);
    }
    if (op_str == std::string("UNLIKELY")) {
      return translateUnlikely(expr, scan_targets);
    }
    if (op_str == std::string("CASE")) {
      return translateCase(expr, scan_targets);
    }
    if (op_str == std::string("ITEM")) {
      return translateItem(expr, scan_targets);
    }
    const auto& operands = expr["operands"];
    CHECK(operands.IsArray());
    if (op_str == std::string("NOW")) {
      CHECK_EQ(unsigned(0), operands.Size());
      return translateNow();
    }
    if (op_str == std::string("DATETIME")) {
      CHECK_EQ(unsigned(1), operands.Size());
      const auto& now_lit = operands[0];
      const std::string datetime_err{R"(Only DATETIME('NOW') supported for now.)"};
      if (!now_lit.IsObject() || !now_lit.HasMember("literal")) {
        throw std::runtime_error(datetime_err);
      }
      const auto now_lit_expr = std::dynamic_pointer_cast<const Analyzer::Constant>(
          translateTypedLiteral(now_lit));
      CHECK(now_lit_expr);
      CHECK(now_lit_expr->get_type_info().is_string());
      if (*now_lit_expr->get_constval().stringval != std::string("NOW")) {
        throw std::runtime_error(datetime_err);
      }
      return translateNow();
    }
    if (op_str == std::string("PG_EXTRACT") || op_str == std::string("PG_DATE_TRUNC")) {
      return translateExtract(
          operands, scan_targets, op_str == std::string("PG_DATE_TRUNC"));
    }
    if (op_str == std::string("DATEADD")) {
      return translateDateadd(operands, scan_targets);
    }
    if (op_str == std::string("DATEDIFF")) {
      return translateDatediff(operands, scan_targets);
    }
    if (op_str == std::string("DATEPART")) {
      return translateDatepart(operands, scan_targets);
    }
    if (op_str == std::string("LENGTH") || op_str == std::string("CHAR_LENGTH")) {
      CHECK_EQ(unsigned(1), operands.Size());
      auto str_arg = getExprFromNode(operands[0], scan_targets);
      return makeExpr<Analyzer::CharLengthExpr>(str_arg->decompress(),
                                                op_str == std::string("CHAR_LENGTH"));
    }
    if (op_str == std::string("$SCALAR_QUERY")) {
      throw std::runtime_error("Subqueries not supported");
    }
    if (operands.Size() == 1) {
      return translateUnaryOp(expr, scan_targets);
    }
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], scan_targets);
    for (size_t i = 1; i < operands.Size(); ++i) {
      std::shared_ptr<Analyzer::Expr> rhs;
      SQLQualifier sql_qual{kONE};
      const auto& rhs_op = operands[i];
      std::tie(rhs, sql_qual) = getQuantifiedRhs(rhs_op, scan_targets);
      if (!rhs) {
        rhs = getExprFromNode(rhs_op, scan_targets);
      }
      CHECK(rhs);
      const auto sql_op = to_sql_op(op_str);
      if (sql_op == kFUNCTION) {
        throw std::runtime_error(std::string("Unsupported operator: ") + op_str);
      }
      lhs = Parser::OperExpr::normalize(sql_op, sql_qual, lhs, rhs);
    }
    return lhs;
  }

  std::shared_ptr<Analyzer::Expr> translateUnaryOp(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_EQ(unsigned(1), operands.Size());
    const auto operand_expr = getExprFromNode(operands[0], scan_targets);
    const auto op_str = expr["op"].GetString();
    const auto sql_op = to_sql_op(op_str);
    switch (sql_op) {
      case kCAST: {
        const auto& expr_type = expr["type"];
        SQLTypeInfo target_ti(to_sql_type(expr_type["type"].GetString()),
                              !expr_type["nullable"].GetBool());
        const auto& operand_ti = operand_expr->get_type_info();
        if (target_ti.is_time() ||
            operand_ti
                .is_string()) {  // TODO(alex): check and unify with the rest of the cases
          return operand_expr->add_cast(target_ti);
        }
        return makeExpr<Analyzer::UOper>(target_ti, false, sql_op, operand_expr);
      }
      case kNOT:
      case kISNULL: {
        return makeExpr<Analyzer::UOper>(kBOOLEAN, sql_op, operand_expr);
      }
      case kISNOTNULL: {
        auto is_null = makeExpr<Analyzer::UOper>(kBOOLEAN, kISNULL, operand_expr);
        return makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
      }
      case kMINUS: {
        const auto& ti = operand_expr->get_type_info();
        return makeExpr<Analyzer::UOper>(ti, false, kUMINUS, operand_expr);
      }
      case kUNNEST: {
        const auto& ti = operand_expr->get_type_info();
        CHECK(ti.is_array());
        return makeExpr<Analyzer::UOper>(
            ti.get_elem_type(), false, kUNNEST, operand_expr);
      }
      default: {
        CHECK(sql_op == kFUNCTION || sql_op == kIN);
        throw std::runtime_error(std::string("Unsupported unary operator: ") + op_str);
      }
    }
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateLike(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets,
      const bool is_ilike) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], scan_targets);
    auto rhs = getExprFromNode(operands[1], scan_targets);
    auto esc = operands.Size() > 2 ? getExprFromNode(operands[2], scan_targets) : nullptr;
    return Parser::LikeExpr::get(lhs, rhs, esc, is_ilike, false);
  }

  std::shared_ptr<Analyzer::Expr> translateRegexp(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], scan_targets);
    auto rhs = getExprFromNode(operands[1], scan_targets);
    auto esc = operands.Size() > 2 ? getExprFromNode(operands[2], scan_targets) : nullptr;
    return Parser::RegexpExpr::get(lhs, rhs, esc, false);
  }

  std::shared_ptr<Analyzer::Expr> translateLikely(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(1));
    auto arg = getExprFromNode(operands[0], scan_targets);
    return Parser::LikelihoodExpr::get(arg, 0.9375, false);
  }

  std::shared_ptr<Analyzer::Expr> translateUnlikely(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(1));
    auto arg = getExprFromNode(operands[0], scan_targets);
    return Parser::LikelihoodExpr::get(arg, 0.0625, false);
  }

  std::shared_ptr<Analyzer::Expr> translateCase(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(2));
    std::shared_ptr<Analyzer::Expr> else_expr;
    std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
        expr_list;
    for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
      const auto when_expr = getExprFromNode(*operands_it++, scan_targets);
      if (operands_it == operands.End()) {
        else_expr = when_expr;
        break;
      }
      const auto then_expr = getExprFromNode(*operands_it++, scan_targets);
      expr_list.emplace_back(when_expr, then_expr);
    }
    return Parser::CaseExpr::normalize(expr_list, else_expr);
  }

  std::shared_ptr<Analyzer::Expr> translateItem(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK(operands.IsArray());
    CHECK_EQ(operands.Size(), unsigned(2));
    auto base = getExprFromNode(operands[0], scan_targets);
    auto index = getExprFromNode(operands[1], scan_targets);
    return makeExpr<Analyzer::BinOper>(
        base->get_type_info().get_elem_type(), false, kARRAY_AT, kONE, base, index);
  }

  std::shared_ptr<Analyzer::Expr> translateNow() {
    return Parser::TimestampLiteral::get(now_);
  }

  std::shared_ptr<Analyzer::Expr> translateExtract(
      const rapidjson::Value& operands,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets,
      const bool is_date_trunc) {
    CHECK(operands.IsArray());
    CHECK_EQ(unsigned(2), operands.Size());
    const auto& timeunit_lit = operands[0];
    if (!timeunit_lit.IsObject() || !timeunit_lit.HasMember("literal")) {
      throw std::runtime_error("The time unit parameter must be a literal.");
    }
    const auto timeunit_lit_expr = std::dynamic_pointer_cast<const Analyzer::Constant>(
        translateTypedLiteral(timeunit_lit));
    const auto from_expr = getExprFromNode(operands[1], scan_targets);
    return is_date_trunc ? Parser::DatetruncExpr::get(
                               from_expr, *timeunit_lit_expr->get_constval().stringval)
                         : Parser::ExtractExpr::get(
                               from_expr, *timeunit_lit_expr->get_constval().stringval);
  }

  std::shared_ptr<Analyzer::Expr> translateDateadd(
      const rapidjson::Value& operands,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    CHECK(operands.IsArray());
    CHECK_EQ(unsigned(3), operands.Size());
    const auto& timeunit_lit = operands[0];
    if (!timeunit_lit.IsObject() || !timeunit_lit.HasMember("literal")) {
      throw std::runtime_error("The time unit parameter must be a literal.");
    }
    const auto timeunit_lit_expr = std::dynamic_pointer_cast<const Analyzer::Constant>(
        translateTypedLiteral(timeunit_lit));
    const auto number_units = getExprFromNode(operands[1], scan_targets);
    const auto datetime = getExprFromNode(operands[2], scan_targets);
    return makeExpr<Analyzer::DateaddExpr>(
        SQLTypeInfo(kTIMESTAMP, false),
        to_dateadd_field(*timeunit_lit_expr->get_constval().stringval),
        number_units,
        datetime);
  }

  std::shared_ptr<Analyzer::Expr> translateDatediff(
      const rapidjson::Value& operands,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    CHECK(operands.IsArray());
    CHECK_EQ(unsigned(3), operands.Size());
    const auto& timeunit_lit = operands[0];
    if (!timeunit_lit.IsObject() || !timeunit_lit.HasMember("literal")) {
      throw std::runtime_error("The time unit parameter must be a literal.");
    }
    const auto timeunit_lit_expr = std::dynamic_pointer_cast<const Analyzer::Constant>(
        translateTypedLiteral(timeunit_lit));
    const auto start = getExprFromNode(operands[1], scan_targets);
    const auto end = getExprFromNode(operands[2], scan_targets);
    return makeExpr<Analyzer::DatediffExpr>(
        SQLTypeInfo(kBIGINT, false),
        to_datediff_field(*timeunit_lit_expr->get_constval().stringval),
        start,
        end);
  }

  std::shared_ptr<Analyzer::Expr> translateDatepart(
      const rapidjson::Value& operands,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    CHECK(operands.IsArray());
    CHECK_EQ(unsigned(2), operands.Size());
    const auto& timeunit_lit = operands[0];
    if (!timeunit_lit.IsObject() || !timeunit_lit.HasMember("literal")) {
      throw std::runtime_error("The time unit parameter must be a literal.");
    }
    const auto timeunit_lit_expr = std::dynamic_pointer_cast<const Analyzer::Constant>(
        translateTypedLiteral(timeunit_lit));
    const auto from_expr = getExprFromNode(operands[1], scan_targets);
    return Parser::ExtractExpr::get(
        from_expr, to_datepart_field(*timeunit_lit_expr->get_constval().stringval));
  }

  std::shared_ptr<Analyzer::Expr> translateColRef(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    int col_name_idx = expr["input"].GetInt();
    CHECK_GE(col_name_idx, 0);
    if (static_cast<size_t>(col_name_idx) < scan_targets.size()) {
      auto var_expr = std::dynamic_pointer_cast<Analyzer::Var>(
          scan_targets[col_name_idx]->get_own_expr());
      if (var_expr) {
        return var_expr;
      }
    }
    int rte_idx{0};
    for (const auto& col_name_td : col_names_) {
      if (static_cast<size_t>(col_name_idx) < col_name_td.names_.size()) {
        const auto& col_name = col_name_td.names_[col_name_idx];
        const auto cd = cat_.getMetadataForColumn(col_name_td.td_->tableId, col_name);
        CHECK(cd);
        used_columns_[col_name_td.td_->tableId].insert(cd->columnId);
        return makeExpr<Analyzer::ColumnVar>(
            cd->columnType, col_name_td.td_->tableId, cd->columnId, rte_idx);
      }
      col_name_idx -= col_name_td.names_.size();
      ++rte_idx;
    }
    CHECK(false);
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateAggregate(
      const rapidjson::Value& expr,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets) {
    CHECK(expr.IsObject() && expr.HasMember("type"));
    const auto& expr_type = expr["type"];
    CHECK(expr_type.IsObject());
    const auto agg_kind = to_agg_kind(expr["agg"].GetString());
    const bool is_distinct = expr["distinct"].GetBool();
    const auto operand = get_agg_operand_idx(expr);
    const bool takes_arg{operand >= 0};
    if (takes_arg) {
      CHECK_LT(operand, static_cast<ssize_t>(scan_targets.size()));
    }
    const auto arg_expr = takes_arg ? scan_targets[operand]->get_own_expr() : nullptr;
    const auto agg_ti = get_agg_type(agg_kind, arg_expr.get());
    return makeExpr<Analyzer::AggExpr>(agg_ti, agg_kind, arg_expr, is_distinct, nullptr);
  }

  std::shared_ptr<Analyzer::Expr> translateTypedLiteral(const rapidjson::Value& expr) {
    const auto parsed_lit = parse_literal(expr);
    const auto& lit_ti = std::get<1>(parsed_lit);
    const auto json_val = std::get<0>(parsed_lit);
    switch (lit_ti.get_type()) {
      case kDECIMAL: {
        CHECK(json_val->IsInt64());
        const auto val = json_val->GetInt64();
        const int precision = lit_ti.get_precision();
        const int scale = lit_ti.get_scale();
        const auto& target_ti = std::get<2>(parsed_lit);
        if (target_ti.is_fp() && !scale) {
          return make_fp_constant(val, target_ti);
        }
        auto lit_expr = scale
                            ? Parser::FixedPtLiteral::analyzeValue(val, scale, precision)
                            : Parser::IntLiteral::analyzeValue(val);
        return scale && lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
      }
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH: {
        CHECK(json_val->IsInt64());
        Datum d;
        d.timeval = json_val->GetInt64();
        return makeExpr<Analyzer::Constant>(lit_ti.get_type(), false, d);
      }
      case kTIME:
      case kTIMESTAMP: {
        CHECK(json_val->IsInt64());
        Datum d;
        d.timeval = json_val->GetInt64() / 1000;
        return makeExpr<Analyzer::Constant>(lit_ti.get_type(), false, d);
      }
      case kDATE: {
        CHECK(json_val->IsInt64());
        Datum d;
        d.timeval = json_val->GetInt64() * 24 * 3600;
        return makeExpr<Analyzer::Constant>(lit_ti.get_type(), false, d);
      }
      case kTEXT: {
        CHECK(json_val->IsString());
        const auto val = json_val->GetString();
        return Parser::StringLiteral::analyzeValue(val);
      }
      case kBOOLEAN: {
        CHECK(json_val->IsBool());
        Datum d;
        d.boolval = json_val->GetBool();
        return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
      }
      case kDOUBLE: {
        CHECK(json_val->IsDouble());
        Datum d;
        d.doubleval = json_val->GetDouble();
        const auto& target_ti = std::get<2>(parsed_lit);
        auto lit_expr = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
        return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
      }
      case kNULLT: {
        const auto& target_ti = std::get<2>(parsed_lit);
        return makeExpr<Analyzer::Constant>(target_ti.get_type(), true, Datum{0});
      }
      default: { LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name(); }
    }
    return nullptr;
  }

  std::list<int> getUsedColumnList(const int32_t table_id) const {
    std::list<int> used_column_list;
    const auto it = used_columns_.find(table_id);
    if (it == used_columns_.end()) {
      return {};
    }
    for (const int used_col : it->second) {
      used_column_list.push_back(used_col);
    }
    return used_column_list;
  }

  std::vector<const TableDescriptor*> getTableDescriptors() const {
    std::vector<const TableDescriptor*> tds;
    for (const auto& col_name_td : col_names_) {
      tds.push_back(col_name_td.td_);
    }
    return tds;
  }

 private:
  static std::vector<std::string> getColNames(const rapidjson::Value& scan_ra) {
    CHECK(scan_ra.IsObject() && scan_ra.HasMember("fieldNames"));
    const auto& col_names_node = scan_ra["fieldNames"];
    CHECK(col_names_node.IsArray());
    std::vector<std::string> result;
    for (auto field_it = col_names_node.Begin(); field_it != col_names_node.End();
         ++field_it) {
      CHECK(field_it->IsString());
      result.emplace_back(field_it->GetString());
    }
    return result;
  }

  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_info = scan_ra["table"];
    CHECK(table_info.IsArray());
    CHECK_EQ(unsigned(3), table_info.Size());
    const auto td = cat_.getMetadataForTable(table_info[2].GetString());
    CHECK(td);
    return td;
  }

  struct ColNames {
    std::vector<std::string> names_;
    const TableDescriptor* td_;
  };

  std::unordered_map<int32_t, std::set<int>> used_columns_;
  const Catalog_Namespace::Catalog& cat_;
  time_t now_;
  std::vector<ColNames> col_names_;
};

void reproject_target_entries(
    std::vector<std::shared_ptr<Analyzer::TargetEntry>>& agg_targets,
    const std::vector<size_t>& result_proj_indices,
    const rapidjson::Value& fields) {
  CHECK(fields.IsArray());
  CHECK_EQ(static_cast<size_t>(fields.Size()), result_proj_indices.size());
  if (result_proj_indices.empty()) {
    return;
  }
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> agg_targets_reproj;
  auto fields_it = fields.Begin();
  for (const auto proj_idx : result_proj_indices) {
    CHECK_LT(proj_idx, agg_targets.size());
    CHECK(fields_it != fields.End());
    CHECK(fields_it->IsString());
    const auto te = agg_targets[proj_idx];
    agg_targets_reproj.emplace_back(new Analyzer::TargetEntry(
        fields_it->GetString(), te->get_own_expr(), te->get_unnest()));
    ++fields_it;
  }
  agg_targets.swap(agg_targets_reproj);
}

struct LogicalSortInfo {
  LogicalSortInfo() : limit(0), offset(0) {}
  int64_t limit;
  int64_t offset;
  std::list<Analyzer::OrderEntry> order_entries;
};

LogicalSortInfo get_logical_sort_info(const rapidjson::Value& rels) {
  LogicalSortInfo result;
  bool found{false};
  for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
    const auto& sort_rel = *rels_it;
    CHECK(sort_rel.IsObject() && sort_rel.HasMember("relOp"));
    if (std::string("LogicalSort") != sort_rel["relOp"].GetString()) {
      continue;
    }
    if (!found) {
      if (sort_rel.HasMember("fetch")) {
        const auto& limit_lit = parse_literal(sort_rel["fetch"]);
        CHECK(std::get<1>(limit_lit).is_decimal() &&
              std::get<1>(limit_lit).get_scale() == 0);
        CHECK(std::get<0>(limit_lit)->IsInt64());
        result.limit = std::get<0>(limit_lit)->GetInt64();
      }
      if (sort_rel.HasMember("offset")) {
        const auto& offset_lit = parse_literal(sort_rel["offset"]);
        CHECK(std::get<1>(offset_lit).is_decimal() &&
              std::get<1>(offset_lit).get_scale() == 0);
        CHECK(std::get<0>(offset_lit)->IsInt64());
        result.offset = std::get<0>(offset_lit)->GetInt64();
      }
      CHECK(sort_rel.HasMember("collation"));
      const auto& collation = sort_rel["collation"];
      CHECK(collation.IsArray());
      for (auto collation_it = collation.Begin(); collation_it != collation.End();
           ++collation_it) {
        const auto& oe_node = *collation_it;
        CHECK(oe_node.IsObject());
        result.order_entries.emplace_back(
            oe_node["field"].GetInt() + 1,
            std::string("DESCENDING") == oe_node["direction"].GetString(),
            std::string("FIRST") == oe_node["nulls"].GetString());
      }
      found = true;
    } else {
      // Looks like there are two structurally identical LogicalSortInfo nodes
      // in the Calcite AST. Validation for now, but maybe they can be different?
      if (sort_rel.HasMember("fetch")) {
        const auto& limit_lit = parse_literal(sort_rel["fetch"]);
        CHECK(std::get<1>(limit_lit).is_decimal() &&
              std::get<1>(limit_lit).get_scale() == 0);
        CHECK(std::get<0>(limit_lit)->IsInt64());
        CHECK_EQ(result.limit, std::get<0>(limit_lit)->GetInt64());
      }
      if (sort_rel.HasMember("offset")) {
        const auto& offset_lit = parse_literal(sort_rel["offset"]);
        CHECK(std::get<1>(offset_lit).is_decimal() &&
              std::get<1>(offset_lit).get_scale() == 0);
        CHECK(std::get<0>(offset_lit)->IsInt64());
        CHECK_EQ(result.offset, std::get<0>(offset_lit)->GetInt64());
      }
      CHECK(sort_rel.HasMember("collation"));
      const auto& collation = sort_rel["collation"];
      CHECK(collation.IsArray());
      CHECK_EQ(static_cast<size_t>(collation.Size()), result.order_entries.size());
      auto oe_it = result.order_entries.begin();
      for (size_t i = 0; i < result.order_entries.size(); ++i, ++oe_it) {
        const auto& oe_node = collation[i];
        const auto& oe = *oe_it;
        CHECK_EQ(oe.is_desc,
                 std::string("DESCENDING") == oe_node["direction"].GetString());
        CHECK_EQ(oe.nulls_first, std::string("FIRST") == oe_node["nulls"].GetString());
      }
    }
  }
  return result;
}

Planner::Scan* get_scan_plan(
    const TableDescriptor* td,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets,
    std::list<std::shared_ptr<Analyzer::Expr>>& q,
    std::list<std::shared_ptr<Analyzer::Expr>>& sq,
    CalciteAdapter& calcite_adapter) {
  return new Planner::Scan(scan_targets,
                           q,
                           0.,
                           nullptr,
                           sq,
                           td->tableId,
                           calcite_adapter.getUsedColumnList(td->tableId));
}

Planner::Plan* get_agg_plan(
    const TableDescriptor* td,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& agg_targets,
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    std::list<std::shared_ptr<Analyzer::Expr>>& q,
    std::list<std::shared_ptr<Analyzer::Expr>>& sq,
    CalciteAdapter& calcite_adapter) {
  Planner::Plan* plan = get_scan_plan(td, scan_targets, q, sq, calcite_adapter);
  if (!agg_targets.empty()) {
    plan = new Planner::AggPlan(agg_targets, 0., plan, groupby_exprs);
  }
  return plan;
}

Planner::Plan* get_sort_plan(
    Planner::Plan* plan,
    const rapidjson::Value& rels,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& scan_targets,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& agg_targets) {
  const auto logical_sort_info = get_logical_sort_info(rels);
  if (!logical_sort_info.order_entries.empty()) {
    const auto& sort_target_entries =
        agg_targets.empty() ? scan_targets : agg_targets;  // TODO(alex)
    plan = new Planner::Sort(
        sort_target_entries, 0, plan, logical_sort_info.order_entries, false);
  }
  return plan;
}

std::vector<size_t> collect_reproject_indices(const rapidjson::Value& exprs) {
  std::vector<size_t> result_proj_indices;
  CHECK(exprs.IsArray());
  for (auto exprs_it = exprs.Begin(); exprs_it != exprs.End(); ++exprs_it) {
    CHECK(exprs_it->IsObject());
    result_proj_indices.push_back((*exprs_it)["input"].GetInt());
  }
  return result_proj_indices;
}

std::vector<std::shared_ptr<Analyzer::TargetEntry>> get_input_targets(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& in_targets,
    const rapidjson::Value& exprs,
    const rapidjson::Value& fields,
    CalciteAdapter& calcite_adapter) {
  CHECK(exprs.IsArray());
  CHECK(fields.IsArray());
  CHECK_EQ(exprs.Size(), fields.Size());
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> result;
  if (in_targets.empty()) {
    auto fields_it = fields.Begin();
    for (auto exprs_it = exprs.Begin(); exprs_it != exprs.End();
         ++exprs_it, ++fields_it) {
      const auto proj_expr = calcite_adapter.getExprFromNode(*exprs_it, in_targets);
      CHECK(fields_it != exprs.End());
      CHECK(fields_it->IsString());
      result.emplace_back(
          new Analyzer::TargetEntry(fields_it->GetString(), proj_expr, false));
    }
  } else {
    result = in_targets;
  }
  return result;
}

bool needs_result_plan(const rapidjson::Value& exprs) {
  for (auto exprs_it = exprs.Begin(); exprs_it != exprs.End(); ++exprs_it) {
    const auto& expr = *exprs_it;
    CHECK(expr.IsObject());
    if (!expr.HasMember("input")) {
      return true;
    }
  }
  return false;
}

std::vector<std::shared_ptr<Analyzer::TargetEntry>> build_var_refs(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& in_targets) {
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> var_refs_to_in_targets;
  for (size_t i = 1; i <= in_targets.size(); ++i) {
    const auto target = in_targets[i - 1];
    var_refs_to_in_targets.emplace_back(new Analyzer::TargetEntry(
        target->get_resname(),
        var_ref(target->get_expr(), Analyzer::Var::kINPUT_OUTER, i),
        false));
  }
  return var_refs_to_in_targets;
}

std::vector<std::shared_ptr<Analyzer::TargetEntry>> build_result_plan_targets(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& in_targets,
    const rapidjson::Value& exprs,
    const rapidjson::Value& fields,
    CalciteAdapter& calcite_adapter) {
  const auto var_refs_to_in_targets = build_var_refs(in_targets);
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> result;
  CHECK(fields.IsArray());
  CHECK_EQ(exprs.Size(), fields.Size());
  auto fields_it = fields.Begin();
  for (auto exprs_it = exprs.Begin(); exprs_it != exprs.End(); ++exprs_it, ++fields_it) {
    const auto analyzer_expr =
        calcite_adapter.getExprFromNode(*exprs_it, var_refs_to_in_targets);
    CHECK(fields_it != exprs.End());
    CHECK(fields_it->IsString());
    result.emplace_back(
        new Analyzer::TargetEntry(fields_it->GetString(), analyzer_expr, false));
  }
  return result;
}

bool targets_are_refs(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& in_targets) {
  CHECK(!in_targets.empty());
  for (const auto target : in_targets) {
    if (!dynamic_cast<const Analyzer::Var*>(target->get_expr())) {
      return false;
    }
  }
  return true;
}

std::vector<std::shared_ptr<Analyzer::TargetEntry>> handle_logical_project(
    std::vector<std::shared_ptr<Analyzer::TargetEntry>>& child_plan_targets,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& in_targets,
    const rapidjson::Value& logical_project,
    CalciteAdapter& calcite_adapter) {
  const auto exprs_mem_it = logical_project.FindMember("exprs");
  CHECK(exprs_mem_it != logical_project.MemberEnd());
  const auto& exprs = exprs_mem_it->value;
  const auto fields_mem_it = logical_project.FindMember("fields");
  CHECK(fields_mem_it != logical_project.MemberEnd());
  const auto& fields = fields_mem_it->value;
  auto result = get_input_targets(in_targets, exprs, fields, calcite_adapter);
  if (in_targets.empty()) {  // source scan was the table itself
    return result;
  }
  // needs a re-projection or a result plan
  if (needs_result_plan(exprs)) {
    if (!targets_are_refs(result)) {
      child_plan_targets = result;
    }
    return build_result_plan_targets(result, exprs, fields, calcite_adapter);
  } else {  // just target permutation. no need to create a result plan
    const auto reproj_indices = collect_reproject_indices(exprs);
    reproject_target_entries(result, reproj_indices, fields);
  }
  return result;
}

std::vector<std::shared_ptr<Analyzer::TargetEntry>> handle_logical_aggregate(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& in_targets,
    std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const rapidjson::Value& logical_aggregate,
    CalciteAdapter& calcite_adapter) {
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> result;
  const auto& agg_nodes = logical_aggregate["aggs"];
  const auto& group_nodes = logical_aggregate["group"];
  CHECK(group_nodes.IsArray());
  for (auto group_nodes_it = group_nodes.Begin(); group_nodes_it != group_nodes.End();
       ++group_nodes_it) {
    CHECK(group_nodes_it->IsInt());
    const int target_idx = group_nodes_it->GetInt();
    groupby_exprs.push_back(set_transient_dict(in_targets[target_idx]->get_own_expr()));
  }
  CHECK(agg_nodes.IsArray());
  const auto& fields = logical_aggregate["fields"];
  CHECK(fields.IsArray());
  auto fields_it = fields.Begin();
  for (auto group_nodes_it = group_nodes.Begin(); group_nodes_it != group_nodes.End();
       ++group_nodes_it, ++fields_it) {
    CHECK(group_nodes_it->IsInt());
    const int target_idx = group_nodes_it->GetInt();
    const auto target = in_targets[target_idx];
    CHECK(fields_it != fields.End());
    CHECK(fields_it->IsString());
    CHECK_EQ(target->get_resname(), fields_it->GetString());
    const auto target_expr = set_transient_dict(target->get_own_expr());
    const auto uoper_expr = dynamic_cast<const Analyzer::UOper*>(target_expr.get());
    const bool is_unnest{uoper_expr && uoper_expr->get_optype() == kUNNEST};
    auto group_var_ref = var_ref(target_expr.get(),
                                 Analyzer::Var::kGROUPBY,
                                 group_nodes_it - group_nodes.Begin() + 1);
    result.emplace_back(
        new Analyzer::TargetEntry(target->get_resname(), group_var_ref, is_unnest));
  }
  for (auto agg_nodes_it = agg_nodes.Begin(); agg_nodes_it != agg_nodes.End();
       ++agg_nodes_it, ++fields_it) {
    auto agg_expr = calcite_adapter.getExprFromNode(*agg_nodes_it, in_targets);
    CHECK(fields_it != fields.End());
    CHECK(fields_it->IsString());
    result.emplace_back(
        new Analyzer::TargetEntry(fields_it->GetString(), agg_expr, false));
  }
  return result;
}

void add_quals(const std::shared_ptr<Analyzer::Expr> qual_expr,
               std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
               std::list<std::shared_ptr<Analyzer::Expr>>& quals) {
  CHECK(qual_expr);
  const auto bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(qual_expr);
  if (!bin_oper) {
    quals.push_back(qual_expr);
    return;
  }
  if (bin_oper->get_optype() == kAND) {
    add_quals(bin_oper->get_own_left_operand(), simple_quals, quals);
    add_quals(bin_oper->get_own_right_operand(), simple_quals, quals);
    return;
  }
  int rte_idx{0};
  const auto simple_qual = bin_oper->normalize_simple_predicate(rte_idx);
  if (simple_qual) {
    simple_quals.push_back(simple_qual);
  } else {
    quals.push_back(qual_expr);
  }
}

void collect_used_columns(std::vector<std::shared_ptr<Analyzer::ColumnVar>>& used_cols,
                          const std::shared_ptr<Analyzer::Expr> expr) {
  const auto col_var = std::dynamic_pointer_cast<Analyzer::ColumnVar>(expr);
  if (col_var && !std::dynamic_pointer_cast<Analyzer::Var>(col_var)) {
    used_cols.push_back(col_var);
    return;
  }
  const auto uoper = std::dynamic_pointer_cast<Analyzer::UOper>(expr);
  if (uoper) {
    collect_used_columns(used_cols, uoper->get_own_operand());
    return;
  }
  const auto bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(expr);
  if (bin_oper) {
    collect_used_columns(used_cols, bin_oper->get_own_left_operand());
    collect_used_columns(used_cols, bin_oper->get_own_right_operand());
    return;
  }
  const auto extract_expr = std::dynamic_pointer_cast<Analyzer::ExtractExpr>(expr);
  if (extract_expr) {
    collect_used_columns(used_cols, extract_expr->get_own_from_expr());
    return;
  }
  const auto datetrunc_expr = std::dynamic_pointer_cast<Analyzer::DatetruncExpr>(expr);
  if (datetrunc_expr) {
    collect_used_columns(used_cols, datetrunc_expr->get_own_from_expr());
    return;
  }
  const auto charlength_expr = std::dynamic_pointer_cast<Analyzer::CharLengthExpr>(expr);
  if (charlength_expr) {
    collect_used_columns(used_cols, charlength_expr->get_own_arg());
    return;
  }
  const auto like_expr = std::dynamic_pointer_cast<Analyzer::LikeExpr>(expr);
  if (like_expr) {
    collect_used_columns(used_cols, like_expr->get_own_arg());
    return;
  }
}

std::unordered_set<int> get_used_table_ids(
    const std::vector<std::shared_ptr<Analyzer::ColumnVar>>& used_cols) {
  std::unordered_set<int> result;
  for (const auto col_var : used_cols) {
    result.insert(col_var->get_table_id());
  }
  return result;
}

void separate_join_quals(
    std::unordered_map<int, std::list<std::shared_ptr<Analyzer::Expr>>>& quals,
    std::list<std::shared_ptr<Analyzer::Expr>>& join_quals,
    const std::list<std::shared_ptr<Analyzer::Expr>>& all_quals) {
  for (auto qual_candidate : all_quals) {
    std::vector<std::shared_ptr<Analyzer::ColumnVar>> used_columns;
    collect_used_columns(used_columns, qual_candidate);
    const auto used_table_ids = get_used_table_ids(used_columns);
    if (used_table_ids.size() > 1) {
      CHECK_EQ(size_t(2), used_table_ids.size());
      join_quals.push_back(qual_candidate);
    } else {
      CHECK(!used_table_ids.empty());
      quals[*used_table_ids.begin()].push_back(qual_candidate);
    }
  }
}

const std::string get_op_name(const rapidjson::Value& obj) {
  CHECK(obj.IsObject());
  const auto field_it = obj.FindMember("relOp");
  CHECK(field_it != obj.MemberEnd());
  const auto& field = field_it->value;
  CHECK(field.IsString());
  return field.GetString();
}

bool match_compound_seq(rapidjson::Value::ConstValueIterator& rels_it,
                        const rapidjson::Value::ConstValueIterator rels_end) {
  auto op_name = get_op_name(*rels_it);
  if (op_name == std::string("LogicalFilter")) {
    ++rels_it;
  }
  op_name = get_op_name(*rels_it++);
  if (op_name != std::string("LogicalProject")) {
    return false;
  }
  if (rels_it == rels_end) {
    return true;
  }
  op_name = get_op_name(*rels_it);
  if (op_name == std::string("LogicalAggregate")) {
    ++rels_it;
    if (rels_it == rels_end) {
      return true;
    }
    op_name = get_op_name(*rels_it);
    if (op_name == std::string("LogicalProject")) {
      ++rels_it;
    }
  }
  return true;
}

bool match_filter_project_seq(rapidjson::Value::ConstValueIterator& rels_it,
                              const rapidjson::Value::ConstValueIterator rels_end) {
  CHECK(rels_it != rels_end);
  auto op_name = get_op_name(*rels_it++);
  if (op_name != std::string("LogicalFilter")) {
    return false;
  }
  if (rels_it != rels_end && get_op_name(*rels_it) == std::string("LogicalProject")) {
    ++rels_it;
  }
  return true;
}

bool match_sort_seq(rapidjson::Value::ConstValueIterator& rels_it,
                    const rapidjson::Value::ConstValueIterator rels_end) {
  auto op_name = get_op_name(*rels_it++);
  if (op_name != std::string("LogicalSort")) {
    return false;
  }
  if (rels_it == rels_end) {
    return true;
  }
  op_name = get_op_name(*rels_it++);
  if (op_name != std::string("LogicalProject")) {
    return false;
  }
  op_name = get_op_name(*rels_it++);
  if (op_name != std::string("LogicalSort")) {
    return false;
  }
  return rels_it == rels_end;
}

// We don't aim to support everything Calcite allows in this adapter. Inspect
// the nodes and reject queries which go beyond the legacy front-end.
bool query_is_supported(const rapidjson::Value& rels) {
  rapidjson::Value::ConstValueIterator rels_it = rels.Begin();
  if (std::string("EnumerableTableScan") != get_op_name(*rels_it++)) {
    return false;
  }
  const auto op_name = get_op_name(*rels_it);
  if (op_name == std::string("EnumerableTableScan")) {
    ++rels_it;
    CHECK(rels_it != rels.End());
    if (get_op_name(*rels_it++) != std::string("LogicalJoin")) {
      return false;
    }
  }
  if (!match_compound_seq(rels_it, rels.End())) {
    return false;
  }
  if (rels_it == rels.End()) {
    return true;
  }
  if (get_op_name(*rels_it) == std::string("LogicalSort")) {
    return match_sort_seq(rels_it, rels.End());
  }
  // HAVING query
  if (!match_filter_project_seq(rels_it, rels.End())) {
    return false;
  }
  if (rels_it == rels.End()) {
    return true;
  }
  if (!match_sort_seq(rels_it, rels.End())) {
    return false;
  }
  return rels_it == rels.End();
}

}  // namespace

Planner::RootPlan* translate_query(const std::string& query,
                                   const Catalog_Namespace::Catalog& cat) {
  rapidjson::Document query_ast;
  query_ast.Parse(query.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  const auto& rels = query_ast["rels"];
  if (!query_is_supported(rels)) {
    throw std::runtime_error("This query is not supported yet");
  }
  CHECK(rels.IsArray());
  CalciteAdapter calcite_adapter(cat, rels);
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> result_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> all_join_simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> all_join_quals;
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> child_res_targets;
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> res_targets;
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  bool is_agg_plan{false};
  bool is_join{false};
  for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
    const auto& crt_node = *rels_it;
    CHECK(crt_node.IsObject());
    const auto rel_op_it = crt_node.FindMember("relOp");
    CHECK(rel_op_it != crt_node.MemberEnd());
    CHECK(rel_op_it->value.IsString());
    if (rel_op_it->value.GetString() == std::string("EnumerableTableScan") ||
        rel_op_it->value.GetString() == std::string("LogicalSort")) {
      continue;
    }
    if (rel_op_it->value.GetString() == std::string("LogicalProject")) {
      res_targets = handle_logical_project(
          child_res_targets, res_targets, crt_node, calcite_adapter);
    } else if (rel_op_it->value.GetString() == std::string("LogicalAggregate")) {
      is_agg_plan = true;
      CHECK(!res_targets.empty());
      res_targets =
          handle_logical_aggregate(res_targets, groupby_exprs, crt_node, calcite_adapter);
    } else if (rel_op_it->value.GetString() == std::string("LogicalFilter")) {
      if (res_targets.empty()) {
        if (is_join) {
          add_quals(calcite_adapter.getExprFromNode(crt_node["condition"], {}),
                    all_join_simple_quals,
                    all_join_quals);
        } else {
          add_quals(calcite_adapter.getExprFromNode(crt_node["condition"], {}),
                    simple_quals,
                    quals);
        }
      } else {
        child_res_targets = res_targets;
        res_targets = build_var_refs(res_targets);
        add_quals(calcite_adapter.getExprFromNode(crt_node["condition"], res_targets),
                  result_quals,
                  result_quals);
      }
    } else if (rel_op_it->value.GetString() == std::string("LogicalJoin")) {
      const auto condition = std::dynamic_pointer_cast<Analyzer::Constant>(
          calcite_adapter.getExprFromNode(crt_node["condition"], {}));
      if (!condition) {
        throw std::runtime_error("Unsupported join condition");
      }
      const auto condition_ti = condition->get_type_info();
      CHECK(condition_ti.is_boolean());
      if (crt_node["joinType"].GetString() != std::string("inner")) {
        throw std::runtime_error("Only inner joins supported for now");
      }
      // TODO(alex): use the information in this node?
      is_join = true;
    } else {
      throw std::runtime_error(std::string("Node ") + rel_op_it->value.GetString() +
                               " not supported yet");
    }
  }
  const auto tds = calcite_adapter.getTableDescriptors();
  if (is_join) {
    CHECK_EQ(size_t(2), tds.size());
  } else {
    CHECK_EQ(size_t(1), tds.size());
  }
  CHECK(!res_targets.empty());
  Planner::Plan* plan{nullptr};
  if (is_join) {
    std::unordered_map<int, std::list<std::shared_ptr<Analyzer::Expr>>> scan_quals;
    std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
    separate_join_quals(scan_quals, join_quals, all_join_quals);
    std::unordered_map<int, std::list<std::shared_ptr<Analyzer::Expr>>> scan_simple_quals;
    std::list<std::shared_ptr<Analyzer::Expr>> join_simple_quals;
    separate_join_quals(scan_simple_quals, join_simple_quals, all_join_simple_quals);
    CHECK_LE(scan_quals.size(), size_t(2));
    CHECK_LE(scan_simple_quals.size(), size_t(2));
    const int outer_tid = tds[0]->tableId;
    const int inner_tid = tds[1]->tableId;
    auto outer_plan = get_agg_plan(tds[0],
                                   {},
                                   {},
                                   groupby_exprs,
                                   scan_quals[outer_tid],
                                   scan_simple_quals[outer_tid],
                                   calcite_adapter);
    auto inner_plan = get_agg_plan(tds[1],
                                   {},
                                   {},
                                   groupby_exprs,
                                   scan_quals[inner_tid],
                                   scan_simple_quals[inner_tid],
                                   calcite_adapter);
    if (child_res_targets.empty()) {
      if (is_agg_plan) {
        plan = new Planner::Join({}, join_quals, 0, outer_plan, inner_plan);
        plan = new Planner::AggPlan(res_targets, 0., plan, groupby_exprs);
      } else {
        plan = new Planner::Join(res_targets, join_quals, 0, outer_plan, inner_plan);
      }
    } else {
      if (is_agg_plan) {
        plan = new Planner::Join({}, join_quals, 0, outer_plan, inner_plan);
        plan = new Planner::AggPlan(child_res_targets, 0., plan, groupby_exprs);
      } else {
        plan =
            new Planner::Join(child_res_targets, join_quals, 0, outer_plan, inner_plan);
      }
      plan = new Planner::Result(res_targets, result_quals, 0, plan, {});
    }
  } else if (child_res_targets.empty()) {
    std::vector<std::shared_ptr<Analyzer::TargetEntry>> agg_targets{
        is_agg_plan ? res_targets
                    : std::vector<std::shared_ptr<Analyzer::TargetEntry>>{}};
    std::vector<std::shared_ptr<Analyzer::TargetEntry>> scan_targets{
        is_agg_plan ? std::vector<std::shared_ptr<Analyzer::TargetEntry>>{}
                    : res_targets};
    plan = get_agg_plan(tds[0],
                        scan_targets,
                        agg_targets,
                        groupby_exprs,
                        quals,
                        simple_quals,
                        calcite_adapter);
  } else {
    std::vector<std::shared_ptr<Analyzer::TargetEntry>> agg_targets{
        is_agg_plan ? child_res_targets
                    : std::vector<std::shared_ptr<Analyzer::TargetEntry>>{}};
    std::vector<std::shared_ptr<Analyzer::TargetEntry>> scan_targets{
        is_agg_plan ? std::vector<std::shared_ptr<Analyzer::TargetEntry>>{}
                    : child_res_targets};
    plan = get_agg_plan(tds[0],
                        scan_targets,
                        agg_targets,
                        groupby_exprs,
                        quals,
                        simple_quals,
                        calcite_adapter);
    plan = new Planner::Result(res_targets, result_quals, 0, plan, {});
  }
  CHECK(plan);
  const auto logical_sort_info = get_logical_sort_info(rels);
  plan = get_sort_plan(plan, rels, {}, res_targets);
  return new Planner::RootPlan(plan,
                               kSELECT,
                               tds[0]->tableId,
                               {},
                               cat,
                               logical_sort_info.limit,
                               logical_sort_info.offset);
}

namespace {

std::string pg_shim_impl(const std::string& query) {
  auto result = query;
  {
    boost::regex unnest_expr{R"((\s+|,)(unnest)\s*\()",
                             boost::regex::extended | boost::regex::icase};
    apply_shim(result, unnest_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), what[1] + "PG_UNNEST(");
    });
  }
  {
    boost::regex cast_true_expr{R"(CAST\s*\(\s*'t'\s+AS\s+boolean\s*\))",
                                boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, cast_true_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "true");
        });
  }
  {
    boost::regex cast_false_expr{R"(CAST\s*\(\s*'f'\s+AS\s+boolean\s*\))",
                                 boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, cast_false_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(), what.length(), "false");
        });
  }
  {
    boost::regex ilike_expr{
        R"((\s+|\()((?!\()[^\s]+)\s+ilike\s+('(?:[^']+|'')+')(\s+escape(\s+('[^']+')))?)",
        boost::regex::perl | boost::regex::icase};
    apply_shim(result, ilike_expr, [](std::string& result, const boost::smatch& what) {
      std::string esc = what[6];
      result.replace(what.position(),
                     what.length(),
                     what[1] + "PG_ILIKE(" + what[2] + ", " + what[3] +
                         (esc.empty() ? "" : ", " + esc) + ")");
    });
  }
  {
    boost::regex regexp_expr{
        R"((\s+)([^\s]+)\s+REGEXP\s+('(?:[^']+|'')+')(\s+escape(\s+('[^']+')))?)",
        boost::regex::perl | boost::regex::icase};
    apply_shim(result, regexp_expr, [](std::string& result, const boost::smatch& what) {
      std::string esc = what[6];
      result.replace(what.position(),
                     what.length(),
                     what[1] + "REGEXP_LIKE(" + what[2] + ", " + what[3] +
                         (esc.empty() ? "" : ", " + esc) + ")");
    });
  }
  {
    boost::regex extract_expr{R"(extract\s*\(\s*(\w+)\s+from\s+(.+)\))",
                              boost::regex::extended | boost::regex::icase};
    apply_shim(result, extract_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(),
                     what.length(),
                     "PG_EXTRACT('" + what[1] + "', " + what[2] + ")");
    });
  }
  {
    boost::regex date_trunc_expr{R"(date_trunc\s*\(\s*(\w+)\s*,(.*)\))",
                                 boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, date_trunc_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(what.position(),
                         what.length(),
                         "PG_DATE_TRUNC('" + what[1] + "', " + what[2] + ")");
        });
  }
  {
    boost::regex quant_expr{R"(\s(any|all)\s+([^(\s|;)]+))",
                            boost::regex::extended | boost::regex::icase};
    apply_shim(result, quant_expr, [](std::string& result, const boost::smatch& what) {
      std::string qual_name = what[1];
      std::string quant_fname{boost::iequals(qual_name, "any") ? "PG_ANY" : "PG_ALL"};
      result.replace(what.position(), what.length(), quant_fname + "(" + what[2] + ")");
    });
  }
  {
    boost::regex immediate_cast_expr{R"(TIMESTAMP\(0\)\s+('[^']+'))",
                                     boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, immediate_cast_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(
              what.position(), what.length(), "CAST(" + what[1] + " AS TIMESTAMP(0))");
        });
  }
  {
    boost::regex immediate_cast_expr{R"(TIMESTAMP\(3\)\s+('[^']+'))",
                                     boost::regex::extended | boost::regex::icase};
    apply_shim(
        result, immediate_cast_expr, [](std::string& result, const boost::smatch& what) {
          result.replace(
              what.position(), what.length(), "CAST(" + what[1] + " AS TIMESTAMP(3))");
        });
  }
  {
    boost::regex corr_expr{R"((\s+|,|\()(corr)\s*\()",
                           boost::regex::extended | boost::regex::icase};
    apply_shim(result, corr_expr, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), what[1] + "CORRELATION(");
    });
  }
  {
    try {
      // the geography regex pattern is expensive and can sometimes run out of stack space
      // on long queries. Treat it separately from the other shims.
      boost::regex cast_to_geography_expr{
          R"(CAST\s*\(\s*(((?!geography).)+)\s+AS\s+geography\s*\))",
          boost::regex::perl | boost::regex::icase};
      apply_shim(result,
                 cast_to_geography_expr,
                 [](std::string& result, const boost::smatch& what) {
                   result.replace(what.position(),
                                  what.length(),
                                  "CastToGeography(" + what[1] + ")");
                 });
    } catch (const std::exception& e) {
      LOG(WARNING) << "Error apply geography cast shim: " << e.what()
                   << "\nContinuing query parse...";
    }
  }
  return result;
}

}  // namespace

std::string pg_shim(const std::string& query) {
  try {
    return pg_shim_impl(query);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Error applying shim: " << e.what() << "\nContinuing query parse...";
    // boost::regex throws an exception about the complexity of matching when
    // the wrong type of quotes are used or they're mismatched. Let the query
    // through unmodified, the parser will throw a much more informative error.
  }
  return query;
}
