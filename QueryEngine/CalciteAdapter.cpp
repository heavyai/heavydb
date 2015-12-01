#include "CalciteAdapter.h"

#include <glog/logging.h>
#include <rapidjson/document.h>

const Planner::RootPlan* translate_query(const std::string& query, const Catalog_Namespace::Catalog& catalog) {
  rapidjson::Document query_ast;
  query_ast.Parse(query.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  CHECK(false);
  return nullptr;
}
