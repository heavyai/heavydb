/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef COLUMN_DESCRIPTOR_H
#define COLUMN_DESCRIPTOR_H

#include <cassert>
#include <string>
#include "../Shared/sqltypes.h"

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

  ColumnDescriptor() : isSystemCol(false), isVirtualCol(false) {}
  ColumnDescriptor(const int tableId, const int columnId, const std::string& columnName, const SQLTypeInfo columnType)
      : tableId(tableId),
        columnId(columnId),
        columnName(columnName),
        sourceName(columnName),
        columnType(columnType),
        isSystemCol(false),
        isVirtualCol(false) {}
};

#endif  // COLUMN_DESCRIPTOR
