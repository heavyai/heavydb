/*
 * Copyright 2021 OmniSci, Inc.
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

#include <unordered_set>

// References table column using 0-based positional index.
struct ColumnByIdxRef {
  int db_id;
  int table_id;
  int column_idx;

  bool operator==(const ColumnByIdxRef& other) const {
    return column_idx == other.column_idx && table_id == other.table_id &&
           db_id == other.db_id;
  }
};

using ColumnByIdxRefSet = std::unordered_set<ColumnByIdxRef>;

namespace std {

template <>
struct hash<ColumnByIdxRef> {
  size_t operator()(const ColumnByIdxRef& col) const {
    return col.db_id ^ col.table_id ^ col.column_idx;
  }
};

}  // namespace std