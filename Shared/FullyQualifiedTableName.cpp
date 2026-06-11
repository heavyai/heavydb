/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
