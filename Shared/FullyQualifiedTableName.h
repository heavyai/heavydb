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

#include <ostream>
#include <string>

namespace shared {
struct FullyQualifiedTableName {
  FullyQualifiedTableName(const std::string& db_name, const std::string& table_name)
      : db_name(db_name), table_name(table_name) {}

  bool operator==(const FullyQualifiedTableName& other) const;

  bool operator!=(const FullyQualifiedTableName& other) const;

  bool operator<(const FullyQualifiedTableName& other) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const FullyQualifiedTableName& table_name);

  size_t hash() const;

  // Required by boost
  friend size_t hash_value(const FullyQualifiedTableName& table_name);

  std::string getSqlReference() const;

  std::string db_name;
  std::string table_name;
};
}  // namespace shared

namespace std {
template <>
struct hash<shared::FullyQualifiedTableName> {
  size_t operator()(const shared::FullyQualifiedTableName& table_name) const {
    return table_name.hash();
  }
};
}  // namespace std
