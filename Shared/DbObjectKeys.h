/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
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

#include <cstddef>
#include <cstdint>

#include <ostream>

#define TRANSIENT_DICT_ID 0
#define TRANSIENT_DICT_DB_ID 0

namespace shared {
struct StringDictKey {
  StringDictKey() : db_id(0), dict_id(0) {}

  StringDictKey(int32_t db_id, int32_t dict_id) : db_id(db_id), dict_id(dict_id) {}

  bool operator==(const StringDictKey& other) const;

  bool operator!=(const StringDictKey& other) const;

  bool operator<(const StringDictKey& other) const;

  friend std::ostream& operator<<(std::ostream& os, const StringDictKey& dict_key);

  size_t hash() const;

  bool isTransientDict() const;

  int32_t db_id;
  int32_t dict_id;
};

struct TableKey {
  TableKey() : db_id(0), table_id(0) {}

  TableKey(int32_t db_id, int32_t table_id) : db_id(db_id), table_id(table_id) {}

  bool operator==(const TableKey& other) const;

  bool operator!=(const TableKey& other) const;

  bool operator<(const TableKey& other) const;

  friend std::ostream& operator<<(std::ostream& os, const TableKey& table_key);

  size_t hash() const;

  int32_t db_id;
  int32_t table_id;
};

struct ColumnKey {
  ColumnKey(int32_t db_id, int32_t table_id, int32_t column_id)
      : db_id(db_id), table_id(table_id), column_id(column_id) {}

  ColumnKey(const TableKey& table_key, int32_t column_id)
      : ColumnKey(table_key.db_id, table_key.table_id, column_id) {}

  bool operator==(const ColumnKey& other) const;

  bool operator!=(const ColumnKey& other) const;

  bool operator<(const ColumnKey& other) const;

  friend std::ostream& operator<<(std::ostream& os, const ColumnKey& column_key);

  size_t hash() const;

  int32_t db_id;
  int32_t table_id;
  int32_t column_id;
};
}  // namespace shared

namespace std {
template <>
struct hash<shared::StringDictKey> {
  size_t operator()(const shared::StringDictKey& dict_key) const {
    return dict_key.hash();
  }
};

template <>
struct hash<shared::TableKey> {
  size_t operator()(const shared::TableKey& table_key) const { return table_key.hash(); }
};

template <>
struct hash<shared::ColumnKey> {
  size_t operator()(const shared::ColumnKey& column_key) const {
    return column_key.hash();
  }
};
}  // namespace std
