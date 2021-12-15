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

#pragma once

#include "RelAlgDagBuilder.h"

#include "SchemaMgr/SchemaProvider.h"

class RelAlgSchemaProvider : public SchemaProvider {
 public:
  RelAlgSchemaProvider(const RelAlgNode& root);

  int getId() const override { return -1; }
  std::string_view getName() const override { return "__RelAlgSchema__"; }

  std::vector<int> listDatabases() const override;
  TableInfoList listTables(int db_id) const override;
  ColumnInfoList listColumns(int db_id, int table_id) const override;

  TableInfoPtr getTableInfo(int db_id, int table_id) const override;
  TableInfoPtr getTableInfo(int db_id, const std::string& table_name) const override;

  ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const override;
  ColumnInfoPtr getColumnInfo(int db_id,
                              int table_id,
                              const std::string& col_name) const override;

 private:
  using TableByNameMap = std::unordered_map<std::string, TableInfoPtr>;
  using ColumnByNameMap = std::unordered_map<std::string, ColumnInfoPtr>;

  TableInfoMap table_infos_;
  std::unordered_map<int, TableByNameMap> table_index_by_name_;
  ColumnInfoMap column_infos_;
  std::unordered_map<TableRef, ColumnByNameMap> column_index_by_name_;
};