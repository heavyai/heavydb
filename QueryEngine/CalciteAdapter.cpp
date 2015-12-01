#include "CalciteAdapter.h"

#include <glog/logging.h>
#include <rapidjson/document.h>

namespace {

std::string get_table_name_from_table_scan(const rapidjson::Value& scan_rel) {
  const auto& table_info = scan_rel["table"];
  CHECK(table_info.IsArray());
  CHECK_EQ(unsigned(3), table_info.Size());
  return table_info[2].GetString();
}

}  // namespace

Planner::RootPlan* translate_query(const std::string& query, const Catalog_Namespace::Catalog& cat) {
  rapidjson::Document query_ast;
  query_ast.Parse(query.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  const auto& rels = query_ast["rels"];
  CHECK(rels.IsArray());
  const auto& scan_rel = rels[0];
  CHECK(scan_rel.IsObject());
  CHECK_EQ(std::string("LogicalTableScan"), scan_rel["relOp"].GetString());
  const auto table_name = get_table_name_from_table_scan(scan_rel);
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  LOG(FATAL) << td->tableName << " " << td->tableId;
  return nullptr;
}
