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

  auto col_descs = get_physical_inputs(&root);
  for (auto& col_desc : col_descs) {
    auto info = col_desc.getColInfo();
    column_infos_.insert(
        std::make_pair(ColumnRef(info->db_id, info->table_id, info->column_id), info));
  }
}

std::vector<int> RelAlgSchemaProvider::listDatabases() const {
  UNREACHABLE();
}

TableInfoMap RelAlgSchemaProvider::listTables(int db_id) const {
  UNREACHABLE();
}

ColumnInfoMap RelAlgSchemaProvider::listColumns(int db_id, int table_id) const {
  UNREACHABLE();
}

TableInfoPtr RelAlgSchemaProvider::getTableInfo(int db_id, int table_id) const {
  auto it = table_infos_.find({db_id, table_id});
  if (it != table_infos_.end()) {
    return it->second;
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