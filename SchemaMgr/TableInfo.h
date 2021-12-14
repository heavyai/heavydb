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

#include "Shared/sqltypes.h"

#include <memory>
#include <unordered_set>

struct TableRef {
  TableRef(int db_id_, int table_id_) : db_id(db_id_), table_id(table_id_) {}

  int db_id;
  int table_id;

  bool operator==(const TableRef& other) const {
    return table_id == other.table_id && db_id == other.db_id;
  }
};

using TableRefSet = std::unordered_set<TableRef>;

struct TableInfo : public TableRef {
  TableInfo(int db_id,
            int table_id,
            const std::string name_,
            int shards_,
            int sharded_column_id_)
      : TableRef(db_id, table_id)
      , name(name_)
      , shards(shards_)
      , sharded_column_id(sharded_column_id_) {}

  std::string name;
  int shards;
  int sharded_column_id;
};

using TableInfoPtr = std::shared_ptr<TableInfo>;
using TableInfoMap = std::unordered_map<TableRef, TableInfoPtr>;

namespace std {

template <>
struct hash<TableRef> {
  size_t operator()(const TableRef& col) const { return col.db_id ^ col.table_id; }
};

}  // namespace std