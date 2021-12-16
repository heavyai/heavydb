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

#include "Catalog.h"
#include "SchemaMgr/SchemaProvider.h"

namespace Catalog_Namespace {

class CatalogSchemaProvider : public SchemaProvider {
 public:
  CatalogSchemaProvider(const Catalog* catalog) : catalog_(catalog) {}
  ~CatalogSchemaProvider() override = default;

  int getId() const override { return 0; }
  std::string_view getName() const override { return "OmniSci"; }

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
  const Catalog* catalog_;
};

}  // namespace Catalog_Namespace