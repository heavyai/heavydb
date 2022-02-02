/*
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

#include "SchemaJson.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

std::string schema_to_json(SchemaProviderPtr schema_provider) {
  auto dbs = schema_provider->listDatabases();
  if (dbs.empty()) {
    return "{}";
  }
  // Current JSON format supports a single database only.
  CHECK_EQ(dbs.size(), 1);
  auto tables = schema_provider->listTables(dbs.front());

  rapidjson::Document doc(rapidjson::kObjectType);

  for (auto tinfo : tables) {
    rapidjson::Value table(rapidjson::kObjectType);
    table.AddMember("name",
                    rapidjson::Value().SetString(rapidjson::StringRef(tinfo->name)),
                    doc.GetAllocator());
    table.AddMember("id", rapidjson::Value().SetInt(tinfo->table_id), doc.GetAllocator());
    table.AddMember(
        "columns", rapidjson::Value(rapidjson::kArrayType), doc.GetAllocator());

    auto columns = schema_provider->listColumns(*tinfo);
    for (const auto& col_info : columns) {
      rapidjson::Value column(rapidjson::kObjectType);
      column.AddMember("name",
                       rapidjson::Value().SetString(rapidjson::StringRef(col_info->name)),
                       doc.GetAllocator());
      column.AddMember(
          "coltype",
          rapidjson::Value().SetInt(static_cast<int>(col_info->type.get_type())),
          doc.GetAllocator());
      column.AddMember(
          "colsubtype",
          rapidjson::Value().SetInt(static_cast<int>(col_info->type.get_subtype())),
          doc.GetAllocator());
      column.AddMember("coldim",
                       rapidjson::Value().SetInt(col_info->type.get_dimension()),
                       doc.GetAllocator());
      column.AddMember("colscale",
                       rapidjson::Value().SetInt(col_info->type.get_scale()),
                       doc.GetAllocator());
      column.AddMember("is_notnull",
                       rapidjson::Value().SetBool(col_info->type.get_notnull()),
                       doc.GetAllocator());
      column.AddMember("is_systemcol",
                       rapidjson::Value().SetBool(col_info->is_rowid),
                       doc.GetAllocator());
      column.AddMember("is_virtualcol",
                       rapidjson::Value().SetBool(col_info->is_rowid),
                       doc.GetAllocator());
      column.AddMember(
          "is_deletedcol", rapidjson::Value().SetBool(false), doc.GetAllocator());
      table["columns"].PushBack(column, doc.GetAllocator());
    }
    doc.AddMember(rapidjson::StringRef(tinfo->name), table, doc.GetAllocator());
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  return std::string(buffer.GetString());
}
