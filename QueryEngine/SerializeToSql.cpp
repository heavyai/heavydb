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

#include "SerializeToSql.h"
#include "ExternalExecutor.h"

ScalarExprToSql::ScalarExprToSql(const RelAlgExecutionUnit* ra_exe_unit,
                                 const Catalog_Namespace::Catalog* catalog)
    : ra_exe_unit_(ra_exe_unit), catalog_(catalog) {}

std::string ScalarExprToSql::visitVar(const Analyzer::Var* var) const {
  auto it = ra_exe_unit_->groupby_exprs.begin();
  std::advance(it, var->get_varno() - 1);
  return visit(it->get());
}

std::string ScalarExprToSql::visitColumnVar(const Analyzer::ColumnVar* col_var) const {
  return serialize_table_ref(col_var->get_table_id(), catalog_) + "." +
         serialize_column_ref(
             col_var->get_table_id(), col_var->get_column_id(), catalog_);
}

std::string ScalarExprToSql::visitConstant(const Analyzer::Constant* constant) const {
  if (constant->get_is_null()) {
    return "NULL";
  }
  const auto& constant_ti = constant->get_type_info();
  const auto result = DatumToString(constant->get_constval(), constant_ti);
  if (constant_ti.is_string()) {
    return "'" + result + "'";
  } else {
    return result;
  }
}

std::string ScalarExprToSql::visitUOper(const Analyzer::UOper* uoper) const {
  const auto operand = uoper->get_operand();
  const auto operand_str = visit(operand);
  const auto optype = uoper->get_optype();
  switch (optype) {
    case kNOT: {
      return "NOT (" + operand_str + ")";
    }
    case kUMINUS: {
      return "-" + operand_str;
    }
    case kISNULL: {
      return operand_str + " IS NULL";
    }
    case kCAST: {
      const auto& operand_ti = operand->get_type_info();
      const auto& target_ti = uoper->get_type_info();
      if (!is_supported_type_for_extern_execution(target_ti)) {
        throw std::runtime_error("Type not supported yet for extern execution: " +
                                 target_ti.get_type_name());
      }
      if ((operand_ti.get_type() == target_ti.get_type()) ||
          ((operand_ti.is_string() && target_ti.is_string()))) {
        return operand_str;
      }
      return "CAST(" + operand_str + " AS " + target_ti.get_type_name() + ")";
    }
    default: {
      throw std::runtime_error("Unary operator type: " + std::to_string(optype) +
                               " not supported");
    }
  }
}

std::string ScalarExprToSql::visitBinOper(const Analyzer::BinOper* bin_oper) const {
  return visit(bin_oper->get_left_operand()) + " " +
         binOpTypeToString(bin_oper->get_optype()) + " " +
         visit(bin_oper->get_right_operand());
}

std::string ScalarExprToSql::visitInValues(const Analyzer::InValues* in_values) const {
  const auto needle = visit(in_values->get_arg());
  const auto haystack = visitList(in_values->get_value_list());
  return needle + " IN (" + boost::algorithm::join(haystack, ", ") + ")";
}

std::string ScalarExprToSql::visitLikeExpr(const Analyzer::LikeExpr* like) const {
  const auto str = visit(like->get_arg());
  const auto pattern = visit(like->get_like_expr());
  const auto result = str + " LIKE " + pattern;
  if (like->get_escape_expr()) {
    const auto escape = visit(like->get_escape_expr());
    return result + " ESCAPE " + escape;
  }
  return result;
}

std::string ScalarExprToSql::visitCaseExpr(const Analyzer::CaseExpr* case_) const {
  std::string case_str = "CASE ";
  const auto& expr_pair_list = case_->get_expr_pair_list();
  for (const auto& expr_pair : expr_pair_list) {
    const auto when = "WHEN " + visit(expr_pair.first.get());
    const auto then = " THEN " + visit(expr_pair.second.get());
    case_str += when + then;
  }
  return case_str + " ELSE " + visit(case_->get_else_expr()) + " END";
}

namespace {

std::string agg_to_string(const Analyzer::AggExpr* agg_expr,
                          const RelAlgExecutionUnit* ra_exe_unit,
                          const Catalog_Namespace::Catalog* catalog) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, catalog);
  const auto agg_type = ::toString(agg_expr->get_aggtype());
  const auto arg =
      agg_expr->get_arg() ? scalar_expr_to_sql.visit(agg_expr->get_arg()) : "*";
  const auto distinct = agg_expr->get_is_distinct() ? "DISTINCT " : "";
  return agg_type + "(" + distinct + arg + ")";
}

}  // namespace

std::string ScalarExprToSql::visitFunctionOper(
    const Analyzer::FunctionOper* func_oper) const {
  std::string result = func_oper->getName();
  if (result == "||") {
    CHECK_EQ(func_oper->getArity(), size_t(2));
    return visit(func_oper->getArg(0)) + "||" + visit(func_oper->getArg(1));
  }
  if (result == "SUBSTRING") {
    result = "SUBSTR";
  }
  std::vector<std::string> arg_strs;
  for (size_t i = 0; i < func_oper->getArity(); ++i) {
    arg_strs.push_back(visit(func_oper->getArg(i)));
  }
  return result + "(" + boost::algorithm::join(arg_strs, ",") + ")";
}

std::string ScalarExprToSql::visitWindowFunction(
    const Analyzer::WindowFunction* window_func) const {
  std::string result = ::toString(window_func->getKind());
  {
    const auto arg_strs = visitList(window_func->getArgs());
    result += "(" + boost::algorithm::join(arg_strs, ",") + ")";
  }
  result += " OVER (";
  {
    const auto partition_strs = visitList(window_func->getPartitionKeys());
    if (!partition_strs.empty()) {
      result += "PARTITION BY " + boost::algorithm::join(partition_strs, ",");
    }
  }
  {
    std::vector<std::string> order_strs;
    const auto& order_keys = window_func->getOrderKeys();
    const auto& collation = window_func->getCollation();
    CHECK_EQ(order_keys.size(), collation.size());
    for (size_t i = 0; i < order_keys.size(); ++i) {
      std::string order_str = visit(order_keys[i].get());
      order_str += collation[i].is_desc ? " DESC" : " ASC";
      // TODO: handle nulls first / last
      order_strs.push_back(order_str);
    }
    if (!order_strs.empty()) {
      result += " ORDER BY " + boost::algorithm::join(order_strs, ",");
    }
  }
  result += ")";
  return result;
}

std::string ScalarExprToSql::visitAggExpr(const Analyzer::AggExpr* agg) const {
  return agg_to_string(agg, ra_exe_unit_, catalog_);
}

std::string ScalarExprToSql::binOpTypeToString(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "=";
    case kNE:
      return "<>";
    case kLT:
      return "<";
    case kLE:
      return "<=";
    case kGT:
      return ">";
    case kGE:
      return ">=";
    case kAND:
      return "AND";
    case kOR:
      return "OR";
    case kMINUS:
      return "-";
    case kPLUS:
      return "+";
    case kMULTIPLY:
      return "*";
    case kDIVIDE:
      return "/";
    case kMODULO:
      return "%";
    case kARRAY_AT:
      return "[]";
    case kOVERLAPS:
      return "OVERLAPS";
    default:
      LOG(FATAL) << "Invalid operator type: " << op_type;
      return "";
  }
}

template <typename List>
std::vector<std::string> ScalarExprToSql::visitList(const List& expressions) const {
  std::vector<std::string> result;
  for (const auto& expression : expressions) {
    result.push_back(visit(expression.get()));
  }
  return result;
}

namespace {

std::string where_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                            const Catalog_Namespace::Catalog* catalog) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, catalog);
  auto qual_strings = scalar_expr_to_sql.visitList(ra_exe_unit->quals);
  const auto simple_qual_strings =
      scalar_expr_to_sql.visitList(ra_exe_unit->simple_quals);
  qual_strings.insert(
      qual_strings.end(), simple_qual_strings.begin(), simple_qual_strings.end());
  return boost::algorithm::join(qual_strings, " AND ");
}

std::string join_condition_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                                     const Catalog_Namespace::Catalog* catalog) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, catalog);
  std::vector<std::string> qual_strings;
  for (const auto& join_level_quals : ra_exe_unit->join_quals) {
    const auto level_qual_strings = scalar_expr_to_sql.visitList(join_level_quals.quals);
    qual_strings.insert(
        qual_strings.end(), level_qual_strings.begin(), level_qual_strings.end());
  }
  return boost::algorithm::join(qual_strings, " AND ");
}

std::string targets_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                              const Catalog_Namespace::Catalog* catalog) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, catalog);
  std::vector<std::string> target_strings;
  for (const auto target : ra_exe_unit->target_exprs) {
    target_strings.push_back(scalar_expr_to_sql.visit(target));
  }
  return boost::algorithm::join(target_strings, ", ");
}

std::string group_by_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                               const Catalog_Namespace::Catalog* catalog) {
  if (ra_exe_unit->groupby_exprs.size() == 1 || !ra_exe_unit->groupby_exprs.front()) {
    return "";
  }
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, catalog);
  const auto group_by_strings = scalar_expr_to_sql.visitList(ra_exe_unit->groupby_exprs);
  return boost::algorithm::join(group_by_strings, ", ");
}

std::string from_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                           const Catalog_Namespace::Catalog* catalog) {
  std::vector<std::string> from_strings;
  for (const auto& input_desc : ra_exe_unit->input_descs) {
    const auto table_ref = serialize_table_ref(input_desc.getTableId(), catalog);
    from_strings.push_back(table_ref);
  }
  return boost::algorithm::join(from_strings, ", ");
}

std::string maybe(const std::string& prefix, const std::string& clause) {
  return clause.empty() ? "" : " " + prefix + " " + clause;
}

}  // namespace

std::string serialize_table_ref(const int table_id,
                                const Catalog_Namespace::Catalog* catalog) {
  if (table_id >= 0) {
    const auto td = catalog->getMetadataForTable(table_id);
    CHECK(td);
    return td->tableName;
  }
  return "\"#temp" + std::to_string(table_id) + "\"";
}

std::string serialize_column_ref(const int table_id,
                                 const int column_id,
                                 const Catalog_Namespace::Catalog* catalog) {
  if (table_id >= 0) {
    const auto cd = catalog->getMetadataForColumn(table_id, column_id);
    CHECK(cd);
    return cd->columnName;
  }
  return "col" + std::to_string(column_id);
}

ExecutionUnitSql serialize_to_sql(const RelAlgExecutionUnit* ra_exe_unit,
                                  const Catalog_Namespace::Catalog* catalog) {
  const auto targets = targets_to_string(ra_exe_unit, catalog);
  const auto from = from_to_string(ra_exe_unit, catalog);
  const auto join_on = join_condition_to_string(ra_exe_unit, catalog);
  const auto where = where_to_string(ra_exe_unit, catalog);
  const auto group = group_by_to_string(ra_exe_unit, catalog);
  return {"SELECT " + targets + " FROM " + from + maybe("ON", join_on) +
              maybe("WHERE", where) + maybe("GROUP BY", group),
          from};
}
