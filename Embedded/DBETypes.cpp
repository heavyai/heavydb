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

#include "DBETypes.h"

namespace EmbeddedDatabase {

/** ColumnDetails methods */

ColumnDetails::ColumnDetails()
    : col_type(ColumnType::UNKNOWN)
    , encoding(ColumnEncoding::NONE)
    , nullable(false)
    , is_array(false)
    , precision(0)
    , scale(0)
    , comp_param(0) {}

ColumnDetails::ColumnDetails(const std::string& _col_name,
                             ColumnType _col_type,
                             ColumnEncoding _encoding,
                             bool _nullable,
                             bool _is_array,
                             int _precision,
                             int _scale,
                             int _comp_param)
    : col_name(_col_name)
    , col_type(_col_type)
    , encoding(_encoding)
    , nullable(_nullable)
    , is_array(_is_array)
    , precision(_precision)
    , scale(_scale)
    , comp_param(_comp_param) {}

/** Row methods */

Row::Row() {}

Row::Row(std::vector<TargetValue>& row) : row_(std::move(row)) {}

int64_t Row::getInt(size_t col_num) {
  if (col_num < row_.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&row_[col_num]);
    const auto value = boost::get<int64_t>(scalar_value);
    return *value;
  }
  return 0;
}

double Row::getDouble(size_t col_num) {
  if (col_num < row_.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&row_[col_num]);
    const auto value = boost::get<double>(scalar_value);
    return *value;
  }
  return 0.;
}

std::string Row::getStr(size_t col_num) {
  if (col_num < row_.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&row_[col_num]);
    auto value = boost::get<NullableString>(scalar_value);
    bool is_null = !value || boost::get<void*>(value);
    if (is_null) {
      return "Empty";
    } else {
      auto value_notnull = boost::get<std::string>(value);
      return *value_notnull;
    }
  }
  return "Out of range";
}

ColumnType sqlToColumnType(const SQLTypes& type) {
  switch (type) {
    case kBOOLEAN:
      return ColumnType::BOOL;
    case kTINYINT:
      return ColumnType::TINYINT;
    case kSMALLINT:
      return ColumnType::SMALLINT;
    case kINT:
      return ColumnType::INT;
    case kBIGINT:
      return ColumnType::BIGINT;
    case kFLOAT:
      return ColumnType::FLOAT;
    case kNUMERIC:
    case kDECIMAL:
      return ColumnType::DECIMAL;
    case kDOUBLE:
      return ColumnType::DOUBLE;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return ColumnType::STR;
    case kTIME:
      return ColumnType::TIME;
    case kTIMESTAMP:
      return ColumnType::TIMESTAMP;
    case kDATE:
      return ColumnType::DATE;
    case kINTERVAL_DAY_TIME:
      return ColumnType::INTERVAL_DAY_TIME;
    case kINTERVAL_YEAR_MONTH:
      return ColumnType::INTERVAL_YEAR_MONTH;
    case kPOINT:
      return ColumnType::POINT;
    case kLINESTRING:
      return ColumnType::LINESTRING;
    case kPOLYGON:
      return ColumnType::POLYGON;
    case kMULTIPOLYGON:
      return ColumnType::MULTIPOLYGON;
    case kGEOMETRY:
      return ColumnType::GEOMETRY;
    case kGEOGRAPHY:
      return ColumnType::GEOGRAPHY;
    default:
      return ColumnType::UNKNOWN;
  }
  return ColumnType::UNKNOWN;
}

ColumnEncoding sqlToColumnEncoding(const EncodingType& type) {
  switch (type) {
    case kENCODING_NONE:
      return ColumnEncoding::NONE;
    case kENCODING_FIXED:
      return ColumnEncoding::FIXED;
    case kENCODING_RL:
      return ColumnEncoding::RL;
    case kENCODING_DIFF:
      return ColumnEncoding::DIFF;
    case kENCODING_DICT:
      return ColumnEncoding::DICT;
    case kENCODING_SPARSE:
      return ColumnEncoding::SPARSE;
    case kENCODING_GEOINT:
      return ColumnEncoding::GEOINT;
    case kENCODING_DATE_IN_DAYS:
      return ColumnEncoding::DATE_IN_DAYS;
    default:
      return ColumnEncoding::NONE;
  }
  return ColumnEncoding::NONE;
}
}  // namespace EmbeddedDatabase