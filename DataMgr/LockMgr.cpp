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

#include "LockMgr.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "QueryEngine/JsonAccessors.h"
#include "gen-cpp/CalciteServer.h"

namespace Lock_Namespace {

using namespace rapidjson;

void getTableNames(std::map<std::string, bool>& tableNames, const std::string query_ra) {
  rapidjson::Document query_ast;
  query_ast.Parse(query_ra.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  getTableNames(tableNames, query_ast);
}

void getTableNames(std::map<std::string, bool>& tableNames, const Value& value) {
  if (value.IsArray()) {
    for (SizeType i = 0; i < value.Size(); ++i) {
      getTableNames(tableNames, value[i]);
    }
    return;
  } else if (value.IsObject()) {
    for (auto mit = value.MemberBegin(); mit != value.MemberEnd(); ++mit) {
      getTableNames(tableNames, mit->value);
    }
  } else {
    return;
  }

  if (value.FindMember("rels") == value.MemberEnd()) {
    return;
  }
  const auto& rels = value["rels"];
  CHECK(rels.IsArray());
  for (SizeType i = 0; i < rels.Size(); ++i) {
    const auto& rel = rels[i];
    const auto& relop = json_str(rel["relOp"]);
    if (rel.FindMember("table") != rel.MemberEnd()) {
      if ("EnumerableTableScan" == relop || "LogicalTableModify" == relop) {
        const auto t = rel["table"].GetArray();
        CHECK(t[1].IsString());
        tableNames[t[1].GetString()] |= "LogicalTableModify" == relop;
      }
    }
  }
}

ChunkKey getTableChunkKey(const Catalog_Namespace::Catalog& cat,
                          const std::string& tableName) {
  if (const auto tdp = cat.getMetadataForTable(tableName, false)) {
    ChunkKey chunk_key{cat.getCurrentDB().dbId, tdp->tableId};
    return chunk_key;
  } else {
    throw std::runtime_error("Table " + tableName + " does not exist.");
  }
}

std::string parse_to_ra(const Catalog_Namespace::Catalog& cat,
                        const std::string& query_str,
                        const Catalog_Namespace::SessionInfo& session_info) {
  return cat.getCalciteMgr()
      .process(session_info, query_str, {}, false, false)
      .plan_result;
}
}  // namespace Lock_Namespace
