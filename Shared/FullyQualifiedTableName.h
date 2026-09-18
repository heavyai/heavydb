/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
