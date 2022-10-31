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

#include "DbObjectKeys.h"

#include <boost/functional/hash.hpp>

#include "Shared/misc.h"

namespace shared {
bool StringDictKey::operator==(const StringDictKey& other) const {
  return db_id == other.db_id && dict_id == other.dict_id;
}

bool StringDictKey::operator!=(const StringDictKey& other) const {
  return !(*this == other);
}

bool StringDictKey::operator<(const StringDictKey& other) const {
  if (db_id != other.db_id) {
    return db_id < other.db_id;
  }
  return dict_id < other.dict_id;
}

std::ostream& operator<<(std::ostream& os, const StringDictKey& dict_key) {
  os << "(db_id: " << dict_key.db_id << ", string_dict_id: " << dict_key.dict_id << ")";
  return os;
}

size_t StringDictKey::hash() const {
  return shared::compute_hash(db_id, dict_id);
}

bool StringDictKey::isTransientDict() const {
  return dict_id == TRANSIENT_DICT_ID;
}

bool TableKey::operator==(const TableKey& other) const {
  return db_id == other.db_id && table_id == other.table_id;
}

bool TableKey::operator!=(const TableKey& other) const {
  return !(*this == other);
}

bool TableKey::operator<(const TableKey& other) const {
  if (db_id != other.db_id) {
    return db_id < other.db_id;
  }
  return table_id < other.table_id;
}

std::ostream& operator<<(std::ostream& os, const TableKey& table_key) {
  os << "(db_id: " << table_key.db_id << ", table_id: " << table_key.table_id << ")";
  return os;
}

size_t TableKey::hash() const {
  return shared::compute_hash(db_id, table_id);
}

size_t hash_value(const TableKey& table_key) {
  return table_key.hash();
}

bool ColumnKey::operator==(const ColumnKey& other) const {
  return db_id == other.db_id && table_id == other.table_id &&
         column_id == other.column_id;
}

bool ColumnKey::operator!=(const ColumnKey& other) const {
  return !(*this == other);
}

bool ColumnKey::operator<(const ColumnKey& other) const {
  if (db_id != other.db_id) {
    return db_id < other.db_id;
  }
  if (table_id != other.table_id) {
    return table_id < other.table_id;
  }
  return column_id < other.column_id;
}

std::ostream& operator<<(std::ostream& os, const ColumnKey& column_key) {
  os << "(db_id: " << column_key.db_id << ", table_id: " << column_key.table_id
     << ", column_id: " << column_key.column_id << ")";
  return os;
}

size_t ColumnKey::hash() const {
  auto hash = shared::compute_hash(table_id, column_id);
  boost::hash_combine(hash, db_id);
  return hash;
}
}  // namespace shared
