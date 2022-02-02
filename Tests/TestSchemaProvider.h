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

#include "SchemaMgr/SchemaProvider.h"

namespace TestHelpers {

class TestSchemaProvider : public SchemaProvider {
 public:
  TestSchemaProvider(int id, const std::string& name) : id_(id), name_(name) {}
  ~TestSchemaProvider() override = default;

  int getId() const override { return id_; }
  std::string_view getName() const override { return name_; }

  std::vector<int> listDatabases() const override {
    std::vector<int> res;
    std::unordered_set<int> ids;
    for (auto& pr : table_index_by_name_) {
      if (!ids.count(pr.first)) {
        res.push_back(pr.first);
        ids.insert(pr.first);
      }
    }
    return res;
  }

  TableInfoList listTables(int db_id) const override {
    TableInfoList res;
    if (table_index_by_name_.count(db_id)) {
      for (auto& pr : table_index_by_name_.at(db_id)) {
        res.push_back(pr.second);
      }
    }
    return res;
  }

  ColumnInfoList listColumns(int db_id, int table_id) const override {
    CHECK_EQ(column_index_by_name_.count({db_id, table_id}), 1);
    auto& table_cols = column_index_by_name_.at({db_id, table_id});
    ColumnInfoList res;
    res.reserve(table_cols.size());
    for (auto [col_name, col_info] : table_cols) {
      res.push_back(col_info);
    }
    std::sort(res.begin(), res.end(), [](ColumnInfoPtr& lhs, ColumnInfoPtr& rhs) -> bool {
      return lhs->column_id < rhs->column_id;
    });
    return res;
  }

  TableInfoPtr getTableInfo(int db_id, int table_id) const override {
    auto it = table_infos_.find({db_id, table_id});
    if (it != table_infos_.end()) {
      return it->second;
    }
    return nullptr;
  }

  TableInfoPtr getTableInfo(int db_id, const std::string& table_name) const override {
    auto db_it = table_index_by_name_.find(db_id);
    if (db_it != table_index_by_name_.end()) {
      auto table_it = db_it->second.find(table_name);
      if (table_it != db_it->second.end()) {
        return table_it->second;
      }
    }
    return nullptr;
  }

  ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const override {
    auto it = column_infos_.find({db_id, table_id, col_id});
    if (it != column_infos_.end()) {
      return it->second;
    }
    return nullptr;
  }

  ColumnInfoPtr getColumnInfo(int db_id,
                              int table_id,
                              const std::string& col_name) const override {
    auto table_it = column_index_by_name_.find({db_id, table_id});
    if (table_it != column_index_by_name_.end()) {
      auto col_it = table_it->second.find(col_name);
      if (col_it != table_it->second.end()) {
        return col_it->second;
      }
    }
    return nullptr;
  }

 protected:
  void addTableInfo(TableInfoPtr table_info) {
    table_infos_[*table_info] = table_info;
    table_index_by_name_[table_info->db_id][table_info->name] = table_info;
  }

  template <typename... Ts>
  void addTableInfo(Ts... args) {
    addTableInfo(std::make_shared<TableInfo>(args...));
  }

  void addColumnInfo(ColumnInfoPtr col_info) {
    column_infos_[*col_info] = col_info;
    column_index_by_name_[{col_info->db_id, col_info->table_id}][col_info->name] =
        col_info;
  }

  template <typename... Ts>
  void addColumnInfo(Ts... args) {
    addColumnInfo(std::make_shared<ColumnInfo>(args...));
  }

  void addRowidColumn(int db_id, int table_id) {
    CHECK_EQ(column_index_by_name_.count({db_id, table_id}), 1);
    int col_id = static_cast<int>(column_index_by_name_[{db_id, table_id}].size() + 1);
    addColumnInfo(db_id, table_id, col_id, "rowid", SQLTypeInfo(SQLTypes::kBIGINT), true);
  }

  using TableByNameMap = std::unordered_map<std::string, TableInfoPtr>;
  using ColumnByNameMap = std::unordered_map<std::string, ColumnInfoPtr>;

  int id_;
  std::string name_;
  TableInfoMap table_infos_;
  std::unordered_map<int, TableByNameMap> table_index_by_name_;
  ColumnInfoMap column_infos_;
  std::unordered_map<TableRef, ColumnByNameMap> column_index_by_name_;
};

}  // namespace TestHelpers
