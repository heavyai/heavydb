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

#include "FullyQualifiedTableName.h"

#include <boost/functional/hash.hpp>

namespace shared {
bool FullyQualifiedTableName::operator==(const FullyQualifiedTableName& other) const {
  return db_name == other.db_name && table_name == other.table_name;
}

bool FullyQualifiedTableName::operator!=(const FullyQualifiedTableName& other) const {
  return !(*this == other);
}

bool FullyQualifiedTableName::operator<(const FullyQualifiedTableName& other) const {
  if (db_name != other.db_name) {
    return db_name < other.db_name;
  }
  return table_name < other.table_name;
}

std::ostream& operator<<(std::ostream& os, const FullyQualifiedTableName& table_name) {
  os << "(db_name: " << table_name.db_name << ", table_name: " << table_name.table_name
     << ")";
  return os;
}

size_t FullyQualifiedTableName::hash() const {
  size_t hash{0};
  boost::hash_combine(hash, db_name);
  boost::hash_combine(hash, table_name);
  return hash;
}

size_t hash_value(const FullyQualifiedTableName& table_name) {
  return table_name.hash();
}

std::string FullyQualifiedTableName::getSqlReference() const {
  return db_name + "." + table_name;
}
}  // namespace shared
