#include "CalciteAdapter.h"

#include "../Parser/ParserNode.h"
#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"

#include <glog/logging.h>
#include <rapidjson/document.h>

#include <set>

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
  CHECK(false);
  return kNULLT;
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
    if (expr.IsInt()) {
      return translateIntLiteral(expr);
    }
    CHECK(false);
    return nullptr;
  }

  std::shared_ptr<Analyzer::Expr> translateOp(const rapidjson::Value& expr, const TableDescriptor* td) {
    const auto op_str = expr["op"].GetString();
    const auto& operands = expr["operands"];
    CHECK(operands.IsArray());
    if (operands.Size() == 1) {
      // TODO(alex): implement the rest, also is it a good idea to handle it here?
      CHECK_EQ(kCAST, to_sql_op(op_str));
      const auto operand_expr = getExprFromNode(operands[0], td, {});
      const auto& expr_type = expr["type"];
      SQLTypeInfo target_ti(to_sql_type(expr_type["type"].GetString()), expr_type["nullable"].GetBool());
      return std::make_shared<Analyzer::UOper>(target_ti, false, kCAST, operand_expr);
    }
    CHECK_GE(operands.Size(), unsigned(2));
    auto lhs = getExprFromNode(operands[0], td, {});
    for (size_t i = 1; i < operands.Size(); ++i) {
      const auto rhs = getExprFromNode(operands[i], td, {});
      lhs = Parser::OperExpr::normalize(to_sql_op(op_str), kONE, lhs, rhs);
    }
    return lhs;
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
    SQLTypeInfo agg_ti(to_sql_type(expr_type["type"].GetString()), expr_type["nullable"].GetBool());
    const auto agg_name = expr["agg"].GetString();
    const auto& agg_operands = expr["operands"];
    CHECK(agg_operands.IsArray());
    CHECK(agg_operands.Size() <= 1);
    size_t operand = agg_operands.Empty() ? 0 : agg_operands[0].GetInt();
    const auto agg_kind = to_agg_kind(agg_name);
    return std::make_shared<Analyzer::AggExpr>(
        agg_ti, agg_kind, agg_kind == kCOUNT ? nullptr : scan_targets[operand]->get_own_expr(), false);
  }

  std::shared_ptr<Analyzer::Expr> translateIntLiteral(const rapidjson::Value& expr) {
    return Parser::IntLiteral::analyzeValue(expr.GetInt64());
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

void collect_target_entries(std::vector<Analyzer::TargetEntry*>& agg_targets,
                            std::vector<Analyzer::TargetEntry*>& scan_targets,
                            const rapidjson::Value& group_nodes,
                            const rapidjson::Value& proj_nodes,
                            const rapidjson::Value& agg_nodes,
                            CalciteAdapter& calcite_adapter,
                            const TableDescriptor* td) {
  CHECK(proj_nodes.IsArray());
  for (size_t i = 0; i < proj_nodes.Size(); ++i) {
    const auto proj_expr = calcite_adapter.getExprFromNode(proj_nodes[i], td, {});
    if (std::dynamic_pointer_cast<const Analyzer::Constant>(proj_expr)) {  // TODO(alex): fix
      continue;
    }
    scan_targets.push_back(new Analyzer::TargetEntry("", proj_expr, false));
    if (!group_nodes.Empty()) {
      agg_targets.push_back(new Analyzer::TargetEntry("", proj_expr, false));
    }
  }
  CHECK(agg_nodes.IsArray());
  for (size_t i = 0; i < agg_nodes.Size(); ++i) {
    auto agg_expr = calcite_adapter.getExprFromNode(agg_nodes[i], td, scan_targets);
    agg_targets.push_back(new Analyzer::TargetEntry("", agg_expr, false));
  }
}

void collect_groupby(const rapidjson::Value& group_nodes,
                     const std::vector<Analyzer::TargetEntry*>& agg_targets,
                     std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs) {
  CHECK(group_nodes.IsArray());
  for (size_t i = 0; i < group_nodes.Size(); ++i) {
    const int target_idx = group_nodes[i].GetInt();
    groupby_exprs.push_back(agg_targets[target_idx]->get_expr()->deep_copy());
  }
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
  std::shared_ptr<Analyzer::Expr> filter_expr;
  if (rels.Size() == 4) {  // TODO(alex)
    const auto& filter_ra = rels[1];
    CHECK(filter_ra.IsObject());
    filter_expr = calcite_adapter.getExprFromNode(filter_ra["condition"], td, {});
  }
  const size_t base_off = rels.Size() == 4 ? 2 : 1;
  const auto& project_ra = rels[base_off];
  const auto& proj_nodes = project_ra["exprs"];
  const auto& agg_nodes = rels[base_off + 1]["aggs"];
  const auto& group_nodes = rels[base_off + 1]["group"];
  CHECK(proj_nodes.IsArray());
  std::vector<Analyzer::TargetEntry*> agg_targets;
  std::vector<Analyzer::TargetEntry*> scan_targets;
  collect_target_entries(agg_targets, scan_targets, group_nodes, proj_nodes, agg_nodes, calcite_adapter, td);
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  collect_groupby(group_nodes, agg_targets, groupby_exprs);
  std::list<std::shared_ptr<Analyzer::Expr>> q;
  std::list<std::shared_ptr<Analyzer::Expr>> sq;
  if (filter_expr) {  // TODO(alex): take advantage of simple qualifiers where possible
    q.push_back(filter_expr);
  }
  auto scan_plan =
      new Planner::Scan(scan_targets, q, 0., nullptr, sq, td->tableId, calcite_adapter.getUsedColumnList());
  auto agg_plan = new Planner::AggPlan(agg_targets, 0., scan_plan, groupby_exprs);
  auto root_plan = new Planner::RootPlan(agg_plan, kSELECT, td->tableId, {}, cat, 0, 0);
  root_plan->print();
  puts("");
  return root_plan;
}
