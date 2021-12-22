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

#include "RelAlgSchemaProvider.h"

RelAlgSchemaProvider::RelAlgSchemaProvider(const RelAlgNode& root) {
  table_infos_ = get_physical_table_infos(&root);
  for (auto& pr : table_infos_) {
    CHECK_EQ(table_index_by_name_[pr.first.db_id].count(pr.second->name), 0);
    table_index_by_name_[pr.first.db_id][pr.second->name] = pr.second;
  }

  column_infos_ = get_physical_column_infos(&root);
  for (auto& [col_ref, col_info] : column_infos_) {
    TableRef table_ref{col_ref.db_id, col_ref.table_id};
    CHECK_EQ(column_index_by_name_[table_ref].count(col_info->name), 0);
    column_index_by_name_[table_ref][col_info->name] = col_info;
  }
}

std::vector<int> RelAlgSchemaProvider::listDatabases() const {
  UNREACHABLE();
}

TableInfoList RelAlgSchemaProvider::listTables(int db_id) const {
  UNREACHABLE();
}

ColumnInfoList RelAlgSchemaProvider::listColumns(int db_id, int table_id) const {
  UNREACHABLE();
}

TableInfoPtr RelAlgSchemaProvider::getTableInfo(int db_id, int table_id) const {
  auto it = table_infos_.find({db_id, table_id});
  if (it != table_infos_.end()) {
    return it->second;
  }
  return nullptr;
}

TableInfoPtr RelAlgSchemaProvider::getTableInfo(int db_id,
                                                const std::string& table_name) const {
  auto db_it = table_index_by_name_.find(db_id);
  if (db_it != table_index_by_name_.end()) {
    auto table_it = db_it->second.find(table_name);
    if (table_it != db_it->second.end()) {
      return table_it->second;
    }
  }
  return nullptr;
}

ColumnInfoPtr RelAlgSchemaProvider::getColumnInfo(int db_id,
                                                  int table_id,
                                                  int col_id) const {
  auto it = column_infos_.find({db_id, table_id, col_id});
  if (it != column_infos_.end()) {
    return it->second;
  }
  return nullptr;
}

ColumnInfoPtr RelAlgSchemaProvider::getColumnInfo(int db_id,
                                                  int table_id,
                                                  const std::string& column_name) const {
  auto table_it = column_index_by_name_.find({db_id, table_id});
  if (table_it != column_index_by_name_.end()) {
    auto col_it = table_it->second.find(column_name);
    if (col_it != table_it->second.end()) {
      return col_it->second;
    }
  }
  return nullptr;
}