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

  int getId() const { return -1; }

  std::vector<int> listDatabases() const override;
  TableInfoMap listTables(int db_id) const override;
  ColumnInfoMap listColumns(int db_id, int table_id) const override;

  TableInfoPtr getTableInfo(int db_id, int table_id) const override;
  ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const override;

 private:
  TableInfoMap table_infos_;
  ColumnInfoMap column_infos_;
};