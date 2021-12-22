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

#include "ColumnInfo.h"
#include "TableInfo.h"

class SchemaProvider {
 public:
  virtual ~SchemaProvider() = default;

  virtual int getId() const = 0;
  virtual std::string_view getName() const = 0;

  virtual std::vector<int> listDatabases() const = 0;

  virtual TableInfoList listTables(int db_id) const = 0;

  virtual ColumnInfoList listColumns(int db_id, int table_id) const = 0;

  ColumnInfoList listColumns(const TableRef& ref) const {
    return listColumns(ref.db_id, ref.table_id);
  }

  virtual TableInfoPtr getTableInfo(int db_id, int table_id) const = 0;
  virtual TableInfoPtr getTableInfo(int db_id, const std::string& table_name) const = 0;

  TableInfoPtr getTableInfo(const TableRef& tref) const {
    return getTableInfo(tref.db_id, tref.table_id);
  }

  virtual ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const = 0;
  virtual ColumnInfoPtr getColumnInfo(int db_id,
                                      int table_id,
                                      const std::string& col_name) const = 0;

  ColumnInfoPtr getColumnInfo(const TableRef& tref, int col_id) const {
    return getColumnInfo(tref.db_id, tref.table_id, col_id);
  }

  ColumnInfoPtr getColumnInfo(const TableRef& tref, const std::string& col_name) const {
    return getColumnInfo(tref.db_id, tref.table_id, col_name);
  }

  ColumnInfoPtr getColumnInfo(const ColumnRef& cref) const {
    return getColumnInfo(cref.db_id, cref.table_id, cref.column_id);
  }
};

using SchemaProviderPtr = std::shared_ptr<SchemaProvider>;