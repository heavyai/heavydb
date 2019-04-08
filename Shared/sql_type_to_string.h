/*
 * Copyright 2018 OmniSci, Inc.
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

#include "sqltypes.h"

#include <string>

inline std::string sql_type_to_str(const SQLTypes& type) {
  switch (type) {
    case kNULLT:
      return "NULL";
    case kBOOLEAN:
      return "BOOL";
    case kCHAR:
      return "CHAR";
    case kVARCHAR:
      return "VARCHAR";
    case kNUMERIC:
      return "NUMERIC";
    case kDECIMAL:
      return "DECIMAL";
    case kINT:
      return "INT";
    case kSMALLINT:
      return "SMALLINT";
    case kFLOAT:
      return "FLOAT";
    case kDOUBLE:
      return "DOUBLE";
    case kTIME:
      return "TIME";
    case kTIMESTAMP:
      return "TIMESTAMP";
    case kBIGINT:
      return "BIGINT";
    case kTEXT:
      return "TEXT";
    case kDATE:
      return "DATE";
    case kARRAY:
      return "ARRAY";
    case kINTERVAL_DAY_TIME:
      return "DAY TIME INTERVAL";
    case kINTERVAL_YEAR_MONTH:
      return "YEAR MONTH INTERVAL";
    case kPOINT:
      return "POINT";
    case kLINESTRING:
      return "LINESTRING";
    case kPOLYGON:
      return "POLYGON";
    case kMULTIPOLYGON:
      return "MULTIPOLYGON";
    case kTINYINT:
      return "TINYINT";
    case kGEOMETRY:
      return "GEOMETRY";
    case kGEOGRAPHY:
      return "GEOGRAPHY";
    case kEVAL_CONTEXT_TYPE:
      return "UNEVALUATED ANY";
    default:
      return "INVALID";
  }
}
