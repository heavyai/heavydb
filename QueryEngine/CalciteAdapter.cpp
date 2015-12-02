#include "CalciteAdapter.h"

#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"

#include <glog/logging.h>
#include <rapidjson/document.h>

#include <set>

namespace {

std::string get_table_name_from_table_scan(const rapidjson::Value& scan_ra) {
  const auto& table_info = scan_ra["table"];
  CHECK(table_info.IsArray());
  CHECK_EQ(unsigned(3), table_info.Size());
  return table_info[2].GetString();
}

class CalciteAdapter {
 public:
  std::shared_ptr<Analyzer::Expr> getExprFromNode(const rapidjson::Value& expr, const TableDescriptor* td) {
    if (expr.IsObject() && expr.HasMember("op")) {
      const auto op_name = expr["op"].GetString();
      if (op_name == std::string(">")) {
        const auto& operands = expr["operands"];
        CHECK(operands.IsArray());
        CHECK_EQ(unsigned(2), operands.Size());
        const auto lhs = getExprFromNode(operands[0], td);
        const auto rhs = getExprFromNode(operands[1], td);
        return std::make_shared<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false), false, kGT, kONE, lhs, rhs);
      }
      CHECK(false);
      return nullptr;
    }
    if (expr.IsObject() && expr.HasMember("input")) {
      const int col_id = expr["input"].GetInt();
      used_columns_.insert(col_id);
      return std::make_shared<Analyzer::ColumnVar>(SQLTypeInfo(kINT, false), td->tableId, col_id, 0);
    }
    if (expr.IsInt()) {
      Datum d;
      d.intval = expr.GetInt();
      return std::make_shared<Analyzer::Constant>(SQLTypeInfo(kINT, false), false, d);
    }
    CHECK(false);
    return nullptr;
  }

  std::list<int> getUsedColumnList() const {
    std::list<int> used_column_list;
    for (const int used_col : used_columns_) {
      used_column_list.insert(used_column_list.end(), used_col);
    }
    return used_column_list;
  }

 private:
  std::set<int> used_columns_;
};

}  // namespace

Planner::RootPlan* translate_query(const std::string& query, const Catalog_Namespace::Catalog& cat) {
  rapidjson::Document query_ast;
  query_ast.Parse(query.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  const auto& rels = query_ast["rels"];
  CHECK(rels.IsArray());
  CHECK_EQ(unsigned(4), rels.Size());
  const auto& scan_ra = rels[0];
  CHECK(scan_ra.IsObject());
  CHECK_EQ(std::string("LogicalTableScan"), scan_ra["relOp"].GetString());
  const auto table_name = get_table_name_from_table_scan(scan_ra);
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  const auto& filter_ra = rels[1];
  CHECK(filter_ra.IsObject());
  CalciteAdapter calcite_adapter;
  const auto filter_expr = calcite_adapter.getExprFromNode(filter_ra["condition"], td);
  std::vector<Analyzer::TargetEntry*> t;
  std::list<std::shared_ptr<Analyzer::Expr>> q;
  std::list<std::shared_ptr<Analyzer::Expr>> sq{filter_expr};
  auto scan_plan = new Planner::Scan(t, q, 0., nullptr, sq, td->tableId, calcite_adapter.getUsedColumnList());
  auto count_expr = std::make_shared<Analyzer::AggExpr>(SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false);
  t.push_back(new Analyzer::TargetEntry("", count_expr, false));
  auto agg_plan = new Planner::AggPlan(t, 0., scan_plan, {});
  auto root_plan = new Planner::RootPlan(agg_plan, kSELECT, td->tableId, {}, cat, 0, 0);
  root_plan->print();
  puts("");
  return root_plan;
}
