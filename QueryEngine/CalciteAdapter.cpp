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
  if (type_name == std::string("CHAR")) {
    return kTEXT;
  }
  if (type_name == std::string("BOOLEAN")) {
    return kBOOLEAN;
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
  CalciteAdapter(const Catalog_Namespace::Catalog& cat, const std::vector<std::string>& col_names)
      : cat_(cat), col_names_(col_names) {}

  std::shared_ptr<Analyzer::Expr> getExprFromNode(const rapidjson::Value& expr,
                                                  const TableDescriptor* td,
                                                  const std::vector<Analyzer::TargetEntry*>& scan_targets) {
    if (expr.IsObject() && expr.HasMember("op")) {
      return translateOp(expr, td);
    }
    if (expr.IsObject() && expr.HasMember("input")) {
      return translateColRef(expr, td);
    }
    if (expr.IsObject() && expr.HasMember("agg")) {
      return translateAggregate(expr, td, scan_targets);
    }
    if (expr.IsObject() && expr.HasMember("literal")) {
      return translateTypedLiteral(expr);
    }
    CHECK(false);
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateOp(const rapidjson::Value& expr, const TableDescriptor* td) {
    const auto op_str = expr["op"].GetString();
    if (op_str == std::string("LIKE")) {
      return translateLike(expr, td);
    }
    const auto& operands = expr["operands"];
    CHECK(operands.IsArray());
    if (operands.Size() == 1) {
      return translateUnaryOp(expr, td);
    }
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], td, {});
    for (size_t i = 1; i < operands.Size(); ++i) {
      const auto rhs = getExprFromNode(operands[i], td, {});
      lhs = Parser::OperExpr::normalize(to_sql_op(op_str), kONE, lhs, rhs);
    }
    return lhs;
  }

  std::shared_ptr<Analyzer::Expr> translateUnaryOp(const rapidjson::Value& expr, const TableDescriptor* td) {
    const auto& operands = expr["operands"];
    CHECK_EQ(unsigned(1), operands.Size());
    const auto operand_expr = getExprFromNode(operands[0], td, {});
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

  std::shared_ptr<Analyzer::Expr> translateLike(const rapidjson::Value& expr, const TableDescriptor* td) {
    const auto& operands = expr["operands"];
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], td, {});
    auto rhs = getExprFromNode(operands[1], td, {});
    auto esc = operands.Size() > 2 ? getExprFromNode(operands[2], td, {}) : nullptr;
    return Parser::LikeExpr::get(lhs, rhs, esc, false, false);
  }

  std::shared_ptr<Analyzer::Expr> translateColRef(const rapidjson::Value& expr, const TableDescriptor* td) {
    const int col_name_idx = expr["input"].GetInt();
    CHECK_GE(col_name_idx, 0);
    CHECK_LT(static_cast<size_t>(col_name_idx), col_names_.size());
    const auto& col_name = col_names_[col_name_idx];
    const auto cd = cat_.getMetadataForColumn(td->tableId, col_name);
    CHECK(cd);
    used_columns_.insert(cd->columnId);
    CHECK(cd);
    return std::make_shared<Analyzer::ColumnVar>(cd->columnType, td->tableId, cd->columnId, 0);
  }

  std::shared_ptr<Analyzer::Expr> translateAggregate(const rapidjson::Value& expr,
                                                     const TableDescriptor* td,
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
      default:
        CHECK(false);
    }
    return nullptr;
  }

  std::list<int> getUsedColumnList() const {
    std::list<int> used_column_list;
    for (const int used_col : used_columns_) {
      used_column_list.push_back(used_col);
    }
    return used_column_list;
  }

  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_info = scan_ra["table"];
    CHECK(table_info.IsArray());
    CHECK_EQ(unsigned(3), table_info.Size());
    const auto td = cat_.getMetadataForTable(table_info[2].GetString());
    CHECK(td);
    return td;
  }

 private:
  std::set<int> used_columns_;
  const Catalog_Namespace::Catalog& cat_;
  const std::vector<std::string> col_names_;
};

const rapidjson::Value* get_first_of_type(const rapidjson::Value& rels, const std::string& type) {
  for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
    const auto& proj_ra = *rels_it;
    if (type == proj_ra["relOp"].GetString()) {
      return &proj_ra;
    }
  }
  return nullptr;
}

void collect_target_entries(std::vector<Analyzer::TargetEntry*>& agg_targets,
                            std::vector<Analyzer::TargetEntry*>& scan_targets,
                            std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                            const rapidjson::Value& proj_nodes,
                            const rapidjson::Value& rels,
                            CalciteAdapter& calcite_adapter,
                            const TableDescriptor* td) {
  CHECK(proj_nodes.IsArray());
  for (auto proj_nodes_it = proj_nodes.Begin(); proj_nodes_it != proj_nodes.End(); ++proj_nodes_it) {
    const auto proj_expr = calcite_adapter.getExprFromNode(*proj_nodes_it, td, {});
    if (std::dynamic_pointer_cast<const Analyzer::Constant>(proj_expr)) {  // TODO(alex): fix
      continue;
    }
    scan_targets.push_back(new Analyzer::TargetEntry("", proj_expr, false));
  }
  const auto agg_ra_ptr = get_first_of_type(rels, "LogicalAggregate");
  if (agg_ra_ptr) {
    const auto& agg_ra = *agg_ra_ptr;
    const auto& agg_nodes = agg_ra["aggs"];
    const auto& group_nodes = agg_ra["group"];
    CHECK(group_nodes.IsArray());
    for (auto group_nodes_it = group_nodes.Begin(); group_nodes_it != group_nodes.End(); ++group_nodes_it) {
      CHECK(group_nodes_it->IsInt());
      const int target_idx = group_nodes_it->GetInt();
      groupby_exprs.push_back(scan_targets[target_idx]->get_expr()->deep_copy());
    }
    CHECK(agg_nodes.IsArray());
    for (auto group_nodes_it = group_nodes.Begin(); group_nodes_it != group_nodes.End(); ++group_nodes_it) {
      CHECK(group_nodes_it->IsInt());
      const int target_idx = group_nodes_it->GetInt();
      agg_targets.push_back(new Analyzer::TargetEntry("", scan_targets[target_idx]->get_own_expr(), false));
    }
    for (auto agg_nodes_it = agg_nodes.Begin(); agg_nodes_it != agg_nodes.End(); ++agg_nodes_it) {
      auto agg_expr = calcite_adapter.getExprFromNode(*agg_nodes_it, td, scan_targets);
      agg_targets.push_back(new Analyzer::TargetEntry("", agg_expr, false));
    }
  }
}

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

std::vector<std::string> get_col_names(const rapidjson::Value& scan_ra) {
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

std::shared_ptr<Analyzer::Expr> get_filter_expr(const rapidjson::Value& rels,
                                                CalciteAdapter& calcite_adapter,
                                                const TableDescriptor* td) {
  CHECK(rels.IsArray());
  for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
    const auto& filter_ra = *rels_it;
    CHECK(filter_ra.IsObject() && filter_ra.HasMember("relOp"));
    if (std::string("LogicalFilter") == filter_ra["relOp"].GetString()) {
      return calcite_adapter.getExprFromNode(filter_ra["condition"], td, {});
    }
  }
  return nullptr;
}

std::vector<size_t> get_result_proj_indices(const rapidjson::Value& rels) {
  CHECK(rels.IsArray() && rels.Size());
  const auto& last_rel = *(rels.End() - 1);
  if (std::string("LogicalProject") != last_rel["relOp"].GetString()) {
    return {};
  }
  const auto first_rel_ptr = get_first_of_type(rels, "LogicalProject");
  if (&last_rel == first_rel_ptr) {
    return {};
  }
  const auto& result_proj_nodes = last_rel["exprs"];
  std::vector<size_t> result_proj_indices;
  CHECK(result_proj_nodes.IsArray());
  for (auto it = result_proj_nodes.Begin(); it != result_proj_nodes.End(); ++it) {
    CHECK(it->IsObject());
    result_proj_indices.push_back((*it)["input"].GetInt());
  }
  return result_proj_indices;
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

}  // namespace

Planner::RootPlan* translate_query(const std::string& query, const Catalog_Namespace::Catalog& cat) {
  rapidjson::Document query_ast;
  query_ast.Parse(query.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  const auto& rels = query_ast["rels"];
  CHECK(rels.IsArray());
  const auto& scan_ra = rels[0];
  CHECK(scan_ra.IsObject());
  CHECK_EQ(std::string("LogicalTableScan"), scan_ra["relOp"].GetString());
  CalciteAdapter calcite_adapter(cat, get_col_names(scan_ra));
  auto td = calcite_adapter.getTableFromScanNode(scan_ra);
  const auto filter_expr = get_filter_expr(rels, calcite_adapter, td);
  const auto project_ra_ptr = get_first_of_type(rels, "LogicalProject");
  CHECK(project_ra_ptr);
  const auto& project_ra = *project_ra_ptr;
  const auto& proj_nodes = project_ra["exprs"];
  CHECK(proj_nodes.IsArray());
  std::vector<Analyzer::TargetEntry*> agg_targets;
  std::vector<Analyzer::TargetEntry*> scan_targets;
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  collect_target_entries(agg_targets, scan_targets, groupby_exprs, proj_nodes, rels, calcite_adapter, td);
  const auto result_proj_indices = get_result_proj_indices(rels);
  reproject_target_entries(agg_targets, result_proj_indices);
  std::list<std::shared_ptr<Analyzer::Expr>> q;
  std::list<std::shared_ptr<Analyzer::Expr>> sq;
  if (filter_expr) {  // TODO(alex): take advantage of simple qualifiers where possible
    q.push_back(filter_expr);
  }
  Planner::Plan* select_plan =
      new Planner::Scan(scan_targets, q, 0., nullptr, sq, td->tableId, calcite_adapter.getUsedColumnList());
  if (!agg_targets.empty()) {
    select_plan = new Planner::AggPlan(agg_targets, 0., select_plan, groupby_exprs);
  }
  const auto logical_sort_info = get_logical_sort_info(rels);
  if (!logical_sort_info.order_entries.empty()) {
    const auto& sort_target_entries = agg_targets.empty() ? scan_targets : agg_targets;
    select_plan = new Planner::Sort(sort_target_entries, 0, select_plan, logical_sort_info.order_entries, false);
  }
  auto root_plan = new Planner::RootPlan(
      select_plan, kSELECT, td->tableId, {}, cat, logical_sort_info.limit, logical_sort_info.offset);
  root_plan->print();
  puts("");
  return root_plan;
}
