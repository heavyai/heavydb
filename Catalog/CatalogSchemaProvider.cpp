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

#include "Catalog/CatalogSchemaProvider.h"
#include "Catalog/Catalog.h"

namespace Catalog_Namespace {

std::vector<int> CatalogSchemaProvider::listDatabases() const {
  return {catalog_->getDatabaseId()};
}

TableInfoList CatalogSchemaProvider::listTables(int db_id) const {
  CHECK_EQ(catalog_->getDatabaseId(), db_id);
  auto tds = catalog_->getAllTableMetadata();

  TableInfoList res;
  res.resize(tds.size());
  for (auto& td : tds) {
    res.emplace_back(catalog_->makeInfo(td));
  }

  return res;
}

ColumnInfoList CatalogSchemaProvider::listColumns(int db_id, int table_id) const {
  CHECK_EQ(catalog_->getDatabaseId(), db_id);
  auto cds = catalog_->getAllColumnMetadataForTable(table_id, true, true, false);

  ColumnInfoList res;
  res.resize(cds.size());
  for (auto& cd : cds) {
    res.emplace_back(cd->makeInfo(db_id));
  }

  return res;
}

TableInfoPtr CatalogSchemaProvider::getTableInfo(int db_id, int table_id) const {
  CHECK_EQ(catalog_->getDatabaseId(), db_id);
  auto td = catalog_->getMetadataForTable(table_id);
  CHECK(td);
  return catalog_->makeInfo(td);
}

TableInfoPtr CatalogSchemaProvider::getTableInfo(int db_id,
                                                 const std::string& table_name) const {
  CHECK_EQ(catalog_->getDatabaseId(), db_id);
  auto td = catalog_->getMetadataForTable(table_name);
  CHECK(td);
  return catalog_->makeInfo(td);
}

ColumnInfoPtr CatalogSchemaProvider::getColumnInfo(int db_id,
                                                   int table_id,
                                                   int col_id) const {
  CHECK_EQ(catalog_->getDatabaseId(), db_id);
  auto cd = catalog_->getMetadataForColumn(table_id, col_id);
  CHECK(cd);
  return cd->makeInfo(db_id);
}

ColumnInfoPtr CatalogSchemaProvider::getColumnInfo(int db_id,
                                                   int table_id,
                                                   const std::string& col_name) const {
  CHECK_EQ(catalog_->getDatabaseId(), db_id);
  auto cd = catalog_->getMetadataForColumn(table_id, col_name);
  CHECK(cd);
  return cd->makeInfo(db_id);
}

}  // namespace Catalog_Namespace
