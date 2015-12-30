#include "CalciteAdapter.h"

#include "../Parser/ParserNode.h"
#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"

#include <glog/logging.h>
#include <rapidjson/document.h>

#include <set>
#include <unordered_map>

namespace {

SQLOps to_sql_op(const std::string& op_str) {
  if (op_str == std::string(">")) {
    return kGT;
  }
  if (op_str == std::string(">=")) {
    return kGE;
  }
  if (op_str == std::string("<")) {
    return kLT;
  }
  if (op_str == std::string("<=")) {
    return kLE;
  }
  if (op_str == std::string("=")) {
    return kEQ;
  }
  if (op_str == std::string("<>")) {
    return kNE;
  }
  if (op_str == std::string("+")) {
    return kPLUS;
  }
  if (op_str == std::string("-")) {
    return kMINUS;
  }
  if (op_str == std::string("*")) {
    return kMULTIPLY;
  }
  if (op_str == std::string("/")) {
    return kDIVIDE;
  }
  if (op_str == "MOD") {
    return kMODULO;
  }
  if (op_str == std::string("AND")) {
    return kAND;
  }
  if (op_str == std::string("OR")) {
    return kOR;
  }
  if (op_str == std::string("CAST")) {
    return kCAST;
  }
  if (op_str == std::string("NOT")) {
    return kNOT;
  }
  if (op_str == std::string("IS NULL")) {
    return kISNULL;
  }
  if (op_str == std::string("IS NOT NULL")) {
    return kISNOTNULL;
  }
  CHECK(false);
  return kEQ;
}

SQLAgg to_agg_kind(const std::string& agg_name) {
  if (agg_name == std::string("COUNT")) {
    return kCOUNT;
  }
  if (agg_name == std::string("MIN")) {
    return kMIN;
  }
  if (agg_name == std::string("MAX")) {
    return kMAX;
  }
  if (agg_name == std::string("SUM")) {
    return kSUM;
  }
  if (agg_name == std::string("AVG")) {
    return kAVG;
  }
  CHECK(false);
  return kCOUNT;
}

SQLTypes to_sql_type(const std::string& type_name) {
  if (type_name == std::string("BIGINT")) {
    return kBIGINT;
  }
  if (type_name == std::string("INTEGER")) {
    return kINT;
  }
  if (type_name == std::string("SMALLINT")) {
    return kSMALLINT;
  }
  if (type_name == std::string("FLOAT")) {
    return kFLOAT;
  }
  if (type_name == std::string("DOUBLE")) {
    return kDOUBLE;
  }
  if (type_name == std::string("DECIMAL")) {
    return kDECIMAL;
  }
  if (type_name == std::string("CHAR") || type_name == std::string("VARCHAR")) {
    return kTEXT;
  }
  if (type_name == std::string("BOOLEAN")) {
    return kBOOLEAN;
  }
  if (type_name == std::string("NULL")) {
    return kNULLT;
  }
  CHECK(false);
  return kNULLT;
}

ssize_t get_agg_operand_idx(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  if (!expr.HasMember("agg")) {
    return -1;
  }
  const auto& agg_operands = expr["operands"];
  CHECK(agg_operands.IsArray());
  CHECK(agg_operands.Size() <= 1);
  return agg_operands.Empty() ? -1 : agg_operands[0].GetInt();
}

std::pair<const rapidjson::Value&, SQLTypeInfo> parse_literal(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  auto val_it = expr.FindMember("literal");
  CHECK(val_it != expr.MemberEnd());
  auto type_it = expr.FindMember("type");
  CHECK(type_it != expr.MemberEnd());
  CHECK(type_it->value.IsString());
  const auto type_name = std::string(type_it->value.GetString());
  auto scale_it = expr.FindMember("scale");
  CHECK(scale_it != expr.MemberEnd());
  CHECK(scale_it->value.IsInt());
  const int scale = scale_it->value.GetInt();
  auto precision_it = expr.FindMember("precision");
  CHECK(precision_it != expr.MemberEnd());
  CHECK(precision_it->value.IsInt());
  const int precision = precision_it->value.GetInt();
  const auto sql_type = to_sql_type(type_name);
  SQLTypeInfo ti(sql_type, 0, 0, false);
  ti.set_scale(scale);
  ti.set_precision(precision);
  return {val_it->value, ti};
}

class CalciteAdapter {
 public:
  CalciteAdapter(const Catalog_Namespace::Catalog& cat, const rapidjson::Value& rels) : cat_(cat) {
    CHECK(rels.IsArray());
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& scan_ra = *rels_it;
      CHECK(scan_ra.IsObject());
      if (scan_ra["relOp"].GetString() != std::string("LogicalTableScan")) {
        break;
      }
      col_names_.emplace_back(ColNames{getColNames(scan_ra), getTableFromScanNode(scan_ra)});
    }
  }

  std::shared_ptr<Analyzer::Expr> getExprFromNode(const rapidjson::Value& expr,
                                                  const std::vector<Analyzer::TargetEntry*>& scan_targets) {
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
    CHECK(false);
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateOp(const rapidjson::Value& expr,
                                              const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    const auto op_str = expr["op"].GetString();
    if (op_str == std::string("LIKE")) {
      return translateLike(expr, scan_targets);
    }
    if (op_str == std::string("CASE")) {
      return translateCase(expr, scan_targets);
    }
    const auto& operands = expr["operands"];
    CHECK(operands.IsArray());
    if (operands.Size() == 1) {
      return translateUnaryOp(expr, scan_targets);
    }
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], scan_targets);
    for (size_t i = 1; i < operands.Size(); ++i) {
      const auto rhs = getExprFromNode(operands[i], scan_targets);
      lhs = Parser::OperExpr::normalize(to_sql_op(op_str), kONE, lhs, rhs);
    }
    return lhs;
  }

  std::shared_ptr<Analyzer::Expr> translateUnaryOp(const rapidjson::Value& expr,
                                                   const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_EQ(unsigned(1), operands.Size());
    const auto operand_expr = getExprFromNode(operands[0], scan_targets);
    const auto op_str = expr["op"].GetString();
    const auto sql_op = to_sql_op(op_str);
    switch (sql_op) {
      case kCAST: {
        const auto& expr_type = expr["type"];
        SQLTypeInfo target_ti(to_sql_type(expr_type["type"].GetString()), expr_type["nullable"].GetBool());
        return std::make_shared<Analyzer::UOper>(target_ti, false, sql_op, operand_expr);
      }
      case kNOT:
      case kISNULL: {
        return std::make_shared<Analyzer::UOper>(kBOOLEAN, sql_op, operand_expr);
      }
      case kISNOTNULL: {
        auto is_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, operand_expr);
        return std::make_shared<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
      }
      case kMINUS: {
        const auto& ti = operand_expr->get_type_info();
        return std::make_shared<Analyzer::UOper>(ti, false, kUMINUS, operand_expr);
      }
      default:
        CHECK(false);
    }
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateLike(const rapidjson::Value& expr,
                                                const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], scan_targets);
    auto rhs = getExprFromNode(operands[1], scan_targets);
    auto esc = operands.Size() > 2 ? getExprFromNode(operands[2], scan_targets) : nullptr;
    return Parser::LikeExpr::get(lhs, rhs, esc, false, false);
  }

  std::shared_ptr<Analyzer::Expr> translateCase(const rapidjson::Value& expr,
                                                const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(2));
    std::shared_ptr<Analyzer::Expr> else_expr;
    std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> expr_list;
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

  std::shared_ptr<Analyzer::Expr> translateColRef(const rapidjson::Value& expr,
                                                  const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    int col_name_idx = expr["input"].GetInt();
    CHECK_GE(col_name_idx, 0);
    if (static_cast<size_t>(col_name_idx) < scan_targets.size()) {
      auto var_expr = std::dynamic_pointer_cast<Analyzer::Var>(scan_targets[col_name_idx]->get_own_expr());
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
        return std::make_shared<Analyzer::ColumnVar>(cd->columnType, col_name_td.td_->tableId, cd->columnId, rte_idx);
      }
      col_name_idx -= col_name_td.names_.size();
      ++rte_idx;
    }
    CHECK(false);
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateAggregate(const rapidjson::Value& expr,
                                                     const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    CHECK(expr.IsObject() && expr.HasMember("type"));
    const auto& expr_type = expr["type"];
    CHECK(expr_type.IsObject());
    const bool is_nullable{expr_type["nullable"].GetBool()};
    SQLTypeInfo agg_ti(to_sql_type(expr_type["type"].GetString()), is_nullable);
    const auto operand = get_agg_operand_idx(expr);
    const auto agg_kind = to_agg_kind(expr["agg"].GetString());
    if (agg_kind == kAVG) {
      agg_ti = SQLTypeInfo(kDOUBLE, is_nullable);
    }
    const bool is_distinct = expr["distinct"].GetBool();
    const bool takes_arg = agg_kind != kCOUNT || is_distinct;
    if (takes_arg) {
      CHECK_GE(operand, ssize_t(0));
      CHECK_LT(operand, static_cast<ssize_t>(scan_targets.size()));
    }
    const auto arg_expr = takes_arg ? scan_targets[operand]->get_own_expr() : nullptr;
    return std::make_shared<Analyzer::AggExpr>(agg_ti, agg_kind, arg_expr, is_distinct);
  }

  std::shared_ptr<Analyzer::Expr> translateTypedLiteral(const rapidjson::Value& expr) {
    const auto parsed_lit = parse_literal(expr);
    const auto sql_type = parsed_lit.second.get_type();
    const auto& json_val = parsed_lit.first;
    const int scale = parsed_lit.second.get_scale();
    const int precision = parsed_lit.second.get_precision();
    switch (sql_type) {
      case kDECIMAL: {
        CHECK(json_val.IsInt64());
        const auto val = json_val.GetInt64();
        return scale ? Parser::FixedPtLiteral::analyzeValue(val, scale, precision)
                     : Parser::IntLiteral::analyzeValue(val);
      }
      case kTEXT: {
        CHECK(json_val.IsString());
        const auto val = json_val.GetString();
        return Parser::StringLiteral::analyzeValue(val);
      }
      case kBOOLEAN: {
        CHECK(json_val.IsBool());
        Datum d;
        d.boolval = json_val.GetBool();
        return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
      }
      case kNULLT: {
        return makeExpr<Analyzer::Constant>(kNULLT, true);
      }
      default:
        CHECK(false);
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
    for (auto field_it = col_names_node.Begin(); field_it != col_names_node.End(); ++field_it) {
      CHECK(field_it->IsString());
      result.push_back(field_it->GetString());
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
  std::vector<ColNames> col_names_;
};

void reproject_target_entries(std::vector<Analyzer::TargetEntry*>& agg_targets,
                              const std::vector<size_t>& result_proj_indices) {
  if (result_proj_indices.empty()) {
    return;
  }
  std::vector<Analyzer::TargetEntry*> agg_targets_reproj;
  for (const auto proj_idx : result_proj_indices) {
    CHECK_LT(proj_idx, agg_targets.size());
    agg_targets_reproj.push_back(agg_targets[proj_idx]);
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
        CHECK(limit_lit.second.is_decimal() && limit_lit.second.get_scale() == 0);
        CHECK(limit_lit.first.IsInt64());
        result.limit = limit_lit.first.GetInt64();
      }
      if (sort_rel.HasMember("offset")) {
        const auto& offset_lit = parse_literal(sort_rel["offset"]);
        CHECK(offset_lit.second.is_decimal() && offset_lit.second.get_scale() == 0);
        CHECK(offset_lit.first.IsInt64());
        result.offset = offset_lit.first.GetInt64();
      }
      CHECK(sort_rel.HasMember("collation"));
      const auto& collation = sort_rel["collation"];
      CHECK(collation.IsArray());
      for (auto collation_it = collation.Begin(); collation_it != collation.End(); ++collation_it) {
        const auto& oe_node = *collation_it;
        CHECK(oe_node.IsObject());
        result.order_entries.emplace_back(oe_node["field"].GetInt() + 1,
                                          std::string("DESCENDING") == oe_node["direction"].GetString(),
                                          std::string("FIRST") == oe_node["nulls"].GetString());
      }
      found = true;
    } else {
      // Looks like there are two structurally identical LogicalSortInfo nodes
      // in the Calcite AST. Validation for now, but maybe they can be different?
      if (sort_rel.HasMember("fetch")) {
        const auto& limit_lit = parse_literal(sort_rel["fetch"]);
        CHECK(limit_lit.second.is_decimal() && limit_lit.second.get_scale() == 0);
        CHECK(limit_lit.first.IsInt64());
        CHECK_EQ(result.limit, limit_lit.first.GetInt64());
      }
      if (sort_rel.HasMember("offset")) {
        const auto& offset_lit = parse_literal(sort_rel["offset"]);
        CHECK(offset_lit.second.is_decimal() && offset_lit.second.get_scale() == 0);
        CHECK(offset_lit.first.IsInt64());
        CHECK_EQ(result.offset, offset_lit.first.GetInt64());
      }
      CHECK(sort_rel.HasMember("collation"));
      const auto& collation = sort_rel["collation"];
      CHECK(collation.IsArray());
      CHECK_EQ(static_cast<size_t>(collation.Size()), result.order_entries.size());
      auto oe_it = result.order_entries.begin();
      for (size_t i = 0; i < result.order_entries.size(); ++i, ++oe_it) {
        const auto& oe_node = collation[i];
        const auto& oe = *oe_it;
        CHECK_EQ(oe.tle_no, oe_node["field"].GetInt() + 1);
        CHECK_EQ(oe.is_desc, std::string("DESCENDING") == oe_node["direction"].GetString());
        CHECK_EQ(oe.nulls_first, std::string("FIRST") == oe_node["nulls"].GetString());
      }
    }
  }
  return result;
}

Planner::Scan* get_scan_plan(const TableDescriptor* td,
                             const std::vector<Analyzer::TargetEntry*>& scan_targets,
                             std::list<std::shared_ptr<Analyzer::Expr>>& q,
                             std::list<std::shared_ptr<Analyzer::Expr>>& sq,
                             CalciteAdapter& calcite_adapter) {
  return new Planner::Scan(
      scan_targets, q, 0., nullptr, sq, td->tableId, calcite_adapter.getUsedColumnList(td->tableId));
}

Planner::Plan* get_agg_plan(const TableDescriptor* td,
                            const std::vector<Analyzer::TargetEntry*>& scan_targets,
                            const std::vector<Analyzer::TargetEntry*>& agg_targets,
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

Planner::Plan* get_sort_plan(Planner::Plan* plan,
                             const rapidjson::Value& rels,
                             const std::vector<Analyzer::TargetEntry*>& scan_targets,
                             const std::vector<Analyzer::TargetEntry*>& agg_targets) {
  const auto logical_sort_info = get_logical_sort_info(rels);
  if (!logical_sort_info.order_entries.empty()) {
    const auto& sort_target_entries = agg_targets.empty() ? scan_targets : agg_targets;  // TODO(alex)
    plan = new Planner::Sort(sort_target_entries, 0, plan, logical_sort_info.order_entries, false);
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

std::vector<Analyzer::TargetEntry*> get_input_targets(const std::vector<Analyzer::TargetEntry*>& in_targets,
                                                      const rapidjson::Value& exprs,
                                                      CalciteAdapter& calcite_adapter) {
  CHECK(exprs.IsArray());
  std::vector<Analyzer::TargetEntry*> result;
  if (in_targets.empty()) {
    for (auto exprs_it = exprs.Begin(); exprs_it != exprs.End(); ++exprs_it) {
      const auto proj_expr = calcite_adapter.getExprFromNode(*exprs_it, in_targets);
      result.push_back(new Analyzer::TargetEntry("", proj_expr, false));
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

std::vector<Analyzer::TargetEntry*> build_var_refs(const std::vector<Analyzer::TargetEntry*>& in_targets) {
  std::vector<Analyzer::TargetEntry*> var_refs_to_in_targets;
  for (size_t i = 1; i <= in_targets.size(); ++i) {
    var_refs_to_in_targets.push_back(new Analyzer::TargetEntry(
        "",
        makeExpr<Analyzer::Var>(in_targets[i - 1]->get_expr()->get_type_info(), Analyzer::Var::kINPUT_OUTER, i),
        false));
  }
  return var_refs_to_in_targets;
}

std::vector<Analyzer::TargetEntry*> build_result_plan_targets(const std::vector<Analyzer::TargetEntry*>& in_targets,
                                                              const rapidjson::Value& exprs,
                                                              CalciteAdapter& calcite_adapter) {
  const auto var_refs_to_in_targets = build_var_refs(in_targets);
  std::vector<Analyzer::TargetEntry*> result;
  for (auto exprs_it = exprs.Begin(); exprs_it != exprs.End(); ++exprs_it) {
    const auto analyzer_expr = calcite_adapter.getExprFromNode(*exprs_it, var_refs_to_in_targets);
    result.push_back(new Analyzer::TargetEntry("", analyzer_expr, false));
  }
  return result;
}

bool targets_are_refs(const std::vector<Analyzer::TargetEntry*>& in_targets) {
  CHECK(!in_targets.empty());
  const bool first_is_ref = dynamic_cast<const Analyzer::Var*>(in_targets.front()->get_expr());
  for (const auto target : in_targets) {
    CHECK_EQ(first_is_ref, !!dynamic_cast<const Analyzer::Var*>(target->get_expr()));
  }
  return first_is_ref;
}

std::vector<Analyzer::TargetEntry*> handle_logical_project(std::vector<Analyzer::TargetEntry*>& child_plan_targets,
                                                           const std::vector<Analyzer::TargetEntry*>& in_targets,
                                                           const rapidjson::Value& logical_project,
                                                           CalciteAdapter& calcite_adapter) {
  const auto exprs_mem_it = logical_project.FindMember("exprs");
  const auto& exprs = exprs_mem_it->value;
  auto result = get_input_targets(in_targets, exprs, calcite_adapter);
  if (in_targets.empty()) {  // source scan was the table itself
    return result;
  }
  // needs a re-projection or a result plan
  if (needs_result_plan(exprs)) {
    if (!targets_are_refs(result)) {
      child_plan_targets = result;
    }
    return build_result_plan_targets(result, exprs, calcite_adapter);
  } else {  // just target permutation. no need to create a result plan
    const auto reproj_indices = collect_reproject_indices(exprs);
    reproject_target_entries(result, reproj_indices);
  }
  return result;
}

std::vector<Analyzer::TargetEntry*> handle_logical_aggregate(const std::vector<Analyzer::TargetEntry*>& in_targets,
                                                             std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                                                             const rapidjson::Value& logical_aggregate,
                                                             CalciteAdapter& calcite_adapter) {
  std::vector<Analyzer::TargetEntry*> result;
  const auto& agg_nodes = logical_aggregate["aggs"];
  const auto& group_nodes = logical_aggregate["group"];
  CHECK(group_nodes.IsArray());
  for (auto group_nodes_it = group_nodes.Begin(); group_nodes_it != group_nodes.End(); ++group_nodes_it) {
    CHECK(group_nodes_it->IsInt());
    const int target_idx = group_nodes_it->GetInt();
    groupby_exprs.push_back(in_targets[target_idx]->get_expr()->deep_copy());
  }
  CHECK(agg_nodes.IsArray());
  for (auto group_nodes_it = group_nodes.Begin(); group_nodes_it != group_nodes.End(); ++group_nodes_it) {
    CHECK(group_nodes_it->IsInt());
    const int target_idx = group_nodes_it->GetInt();
    result.push_back(new Analyzer::TargetEntry("", in_targets[target_idx]->get_own_expr(), false));
  }
  for (auto agg_nodes_it = agg_nodes.Begin(); agg_nodes_it != agg_nodes.End(); ++agg_nodes_it) {
    auto agg_expr = calcite_adapter.getExprFromNode(*agg_nodes_it, in_targets);
    result.push_back(new Analyzer::TargetEntry("", agg_expr, false));
  }
  return result;
}

}  // namespace

Planner::RootPlan* translate_query(const std::string& query, const Catalog_Namespace::Catalog& cat) {
  rapidjson::Document query_ast;
  query_ast.Parse(query.c_str());
  query_ast.Parse(query.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  const auto& rels = query_ast["rels"];
  CHECK(rels.IsArray());
  CalciteAdapter calcite_adapter(cat, rels);
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> result_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
  std::vector<Analyzer::TargetEntry*> child_res_targets;
  std::vector<Analyzer::TargetEntry*> res_targets;
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  bool is_agg_plan{false};
  bool is_join{false};
  for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
    const auto& crt_node = *rels_it;
    CHECK(crt_node.IsObject());
    const auto rel_op_it = crt_node.FindMember("relOp");
    CHECK(rel_op_it != crt_node.MemberEnd());
    CHECK(rel_op_it->value.IsString());
    if (rel_op_it->value.GetString() == std::string("LogicalTableScan") ||
        rel_op_it->value.GetString() == std::string("LogicalSort")) {
      continue;
    }
    if (rel_op_it->value.GetString() == std::string("LogicalProject")) {
      res_targets = handle_logical_project(child_res_targets, res_targets, crt_node, calcite_adapter);
    } else if (rel_op_it->value.GetString() == std::string("LogicalAggregate")) {
      is_agg_plan = true;
      CHECK(!res_targets.empty());
      res_targets = handle_logical_aggregate(res_targets, groupby_exprs, crt_node, calcite_adapter);
    } else if (rel_op_it->value.GetString() == std::string("LogicalFilter")) {
      if (res_targets.empty()) {
        if (is_join) {
          join_quals.push_back(calcite_adapter.getExprFromNode(crt_node["condition"], {}));
        } else {
          quals.push_back(calcite_adapter.getExprFromNode(crt_node["condition"], {}));
        }
      } else {
        child_res_targets = res_targets;
        res_targets = build_var_refs(res_targets);
        result_quals.push_back(calcite_adapter.getExprFromNode(crt_node["condition"], res_targets));
      }
    } else if (rel_op_it->value.GetString() == std::string("LogicalJoin")) {
      // TODO(alex): use the information in this node?
      is_join = true;
    } else {
      CHECK(false);
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
    // TODO(alex): properly build the outer and inner plans
    auto outer_plan = get_agg_plan(tds[0], {}, {}, groupby_exprs, quals, simple_quals, calcite_adapter);
    auto inner_plan = get_agg_plan(tds[1], {}, {}, groupby_exprs, quals, simple_quals, calcite_adapter);
    plan = new Planner::Join({}, join_quals, 0, outer_plan, inner_plan);
    if (is_agg_plan) {
      plan = new Planner::AggPlan(res_targets, 0., plan, groupby_exprs);
    } else {
      CHECK(false);
    }
  } else if (child_res_targets.empty()) {
    std::vector<Analyzer::TargetEntry*> agg_targets{is_agg_plan ? res_targets : std::vector<Analyzer::TargetEntry*>{}};
    std::vector<Analyzer::TargetEntry*> scan_targets{is_agg_plan ? std::vector<Analyzer::TargetEntry*>{} : res_targets};
    plan = get_agg_plan(tds[0], scan_targets, agg_targets, groupby_exprs, quals, simple_quals, calcite_adapter);
  } else {
    std::vector<Analyzer::TargetEntry*> agg_targets{is_agg_plan ? child_res_targets
                                                                : std::vector<Analyzer::TargetEntry*>{}};
    std::vector<Analyzer::TargetEntry*> scan_targets{is_agg_plan ? std::vector<Analyzer::TargetEntry*>{}
                                                                 : child_res_targets};
    plan = get_agg_plan(tds[0], scan_targets, agg_targets, groupby_exprs, quals, simple_quals, calcite_adapter);
    plan = new Planner::Result(res_targets, result_quals, 0, plan, {});
  }
  CHECK(plan);
  const auto logical_sort_info = get_logical_sort_info(rels);
  plan = get_sort_plan(plan, rels, {}, res_targets);
  auto root_plan =
      new Planner::RootPlan(plan, kSELECT, tds[0]->tableId, {}, cat, logical_sort_info.limit, logical_sort_info.offset);
  root_plan->print();
  puts("");
  return root_plan;
}
