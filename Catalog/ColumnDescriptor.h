/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COLUMN_DESCRIPTOR_H
#define COLUMN_DESCRIPTOR_H

#include <cassert>
#include <optional>
#include <string>
#include "../Shared/StringTransform.h"
#include "../Shared/sqltypes.h"
#include "../Shared/toString.h"

/**
 * @type ColumnDescriptor
 * @brief specifies the content in-memory of a row in the column metadata table
 *
 */

struct ColumnDescriptor {
  int tableId;
  int columnId;
  std::string columnName;
  std::string sourceName;
  SQLTypeInfo columnType;
  std::string chunks;
  bool isSystemCol;
  bool isVirtualCol;
  std::string virtualExpr;
  bool isDeletedCol;
  bool isGeoPhyCol{false};
  std::optional<std::string> default_value;
  int32_t db_id;
  std::optional<std::string> comment;

  ColumnDescriptor() : isSystemCol(false), isVirtualCol(false), isDeletedCol(false) {}
  ColumnDescriptor(const int tableId,
                   const int columnId,
                   const std::string& columnName,
                   const SQLTypeInfo columnType,
                   int32_t db_id)
      : tableId(tableId)
      , columnId(columnId)
      , columnName(columnName)
      , sourceName(columnName)
      , columnType(columnType)
      , isSystemCol(false)
      , isVirtualCol(false)
      , isDeletedCol(false)
      , db_id(db_id) {}
  ColumnDescriptor(const bool isGeoPhyCol) : ColumnDescriptor() {
    this->isGeoPhyCol = isGeoPhyCol;
  }

  std::string toString() const {
    return ::typeName(this) + "(tableId=" + ::toString(tableId) +
           ", columnId=" + ::toString(columnId) +
           ", columnName=" + ::toString(columnName) +
           ", columnType=" + ::toString(columnType) +
           ", defaultValue=" + ::toString(default_value) +
           ", db_id=" + ::toString(db_id) + ")";
  }

  std::string getDefaultValueLiteral() const {
    // some preprocessing of strings, arrays and especially arrays of strings
    CHECK(default_value.has_value());
    if (columnType.is_string() || columnType.is_geometry() || columnType.is_time()) {
      return "\'" + default_value.value() + "\'";
    } else if (columnType.is_array()) {
      auto value = default_value.value();
      CHECK(value.front() == '{' && value.back() == '}');
      value = value.substr(1, value.length() - 2);
      if (columnType.is_string_array() || is_datetime(columnType.get_subtype())) {
        auto elements = split(value, ", ");
        value = "ARRAY[";
        for (size_t i = 0; i < elements.size(); ++i) {
          value += "'" + elements[i] + "'";
          if (i != elements.size() - 1) {
            value += ", ";
          }
        }
        value += "]";
      } else {
        value = "ARRAY[" + value + "]";
      }
      return value;
    } else {
      return default_value.value();
    }
  }
};

#endif  // COLUMN_DESCRIPTOR
