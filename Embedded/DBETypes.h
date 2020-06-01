/*
 * Copyright 2020 OmniSci, Inc.
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

#include "QueryEngine/TargetValue.h"
#include "Shared/sqltypes.h"

namespace EmbeddedDatabase {

enum class ColumnType {
  SMALLINT = 0,
  INT = 1,
  BIGINT = 2,
  FLOAT = 3,
  DECIMAL = 4,
  DOUBLE = 5,
  STR = 6,
  TIME = 7,
  TIMESTAMP = 8,
  DATE = 9,
  BOOL = 10,
  INTERVAL_DAY_TIME = 11,
  INTERVAL_YEAR_MONTH = 12,
  POINT = 13,
  LINESTRING = 14,
  POLYGON = 15,
  MULTIPOLYGON = 16,
  TINYINT = 17,
  GEOMETRY = 18,
  GEOGRAPHY = 19,
  UNKNOWN = 20
};

enum class ColumnEncoding {
  NONE = 0,
  FIXED = 1,
  RL = 2,
  DIFF = 3,
  DICT = 4,
  SPARSE = 5,
  GEOINT = 6,
  DATE_IN_DAYS = 7
};

class ColumnDetails {
 public:
  std::string col_name;
  ColumnType col_type;
  ColumnEncoding encoding;
  bool nullable;
  bool is_array;
  int precision;
  int scale;
  int comp_param;

  ColumnDetails();

  ColumnDetails(const std::string& col_name,
                ColumnType col_type,
                ColumnEncoding encoding,
                bool nullable,
                bool is_array,
                int precision,
                int scale,
                int comp_param);
};

class Row {
 public:
  Row();
  Row(std::vector<TargetValue>& row);
  int64_t getInt(size_t col_num);
  double getDouble(size_t col_num);
  std::string getStr(size_t col_num);

 private:
  std::vector<TargetValue> row_;
};

ColumnType sqlToColumnType(const SQLTypes& type);
ColumnEncoding sqlToColumnEncoding(const EncodingType& type);
}  // namespace EmbeddedDatabase
