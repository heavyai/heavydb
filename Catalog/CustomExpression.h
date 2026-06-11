/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace Catalog_Namespace {
enum class DataSourceType { TABLE = 0 };

struct CustomExpression {
  CustomExpression(const std::string& name,
                   const std::string& expression_json,
                   DataSourceType data_source_type,
                   int32_t data_source_id)
      : name(name)
      , expression_json(expression_json)
      , data_source_type(data_source_type)
      , data_source_id(data_source_id) {}

  CustomExpression(int32_t id,
                   const std::string& name,
                   const std::string& expression_json,
                   DataSourceType data_source_type,
                   int32_t data_source_id,
                   bool is_deleted)
      : id(id)
      , name(name)
      , expression_json(expression_json)
      , data_source_type(data_source_type)
      , data_source_id(data_source_id)
      , is_deleted(is_deleted) {}

  static std::string dataSourceTypeToString(DataSourceType type_enum) {
    // Only table data source type is currently supported
    CHECK(type_enum == DataSourceType::TABLE)
        << "Unexpected data source type: " << static_cast<int>(type_enum);
    return "TABLE";
  }

  static DataSourceType dataSourceTypeFromString(const std::string& type_str) {
    // Only table data source type is currently supported
    CHECK_EQ(type_str, "TABLE");
    return DataSourceType::TABLE;
  }

  int32_t id{-1};
  std::string name;
  std::string expression_json;
  DataSourceType data_source_type;
  int32_t data_source_id{-1};
  bool is_deleted{false};
};
}  // namespace Catalog_Namespace
