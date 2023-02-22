/*
 * Copyright 2023 HEAVY.AI, Inc.
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

/*
  Provides a light-weight data structure SQLTypeInfoLite to serialize
  SQLTypeInfo (from sqltypes.h) for the extension functions (in
  heavydbTypes.h) by FlatBufferManager.

  Extend SQLTypeInfoLite struct as needed but keep it simple so that
  both sqltypes.h and heavydbTypes.h are able to include it (recall,
  the two header files cannot include each other).
*/

#pragma once

struct SQLTypeInfoLite {
  enum SQLTypes {
    UNSPECIFIED = 0,
    BOOLEAN,
    TINYINT,
    SMALLINT,
    INT,
    BIGINT,
    FLOAT,
    DOUBLE,
    POINT,
    LINESTRING,
    POLYGON,
    MULTIPOINT,
    MULTILINESTRING,
    MULTIPOLYGON,
    TEXT,
    ARRAY
  };
  enum EncodingType {
    NONE = 0,
    DICT,   // used by TEXT and ARRAY of TEXT
    GEOINT  // used by geotypes
  };
  SQLTypes type;
  SQLTypes subtype;          // used by ARRAY
  EncodingType compression;  // used by geotypes and TEXT and ARRAY of TEXT
  int32_t dimension;         // used by geotypes (input_srid)
  int32_t scale;             // used by geotypes (output_srid)
  int32_t db_id;             // used by TEXT and ARRAY of TEXT
  int32_t dict_id;           // used by TEXT and ARRAY of TEXT

  inline bool is_geoint() const { return compression == GEOINT; }
  inline int32_t get_input_srid() const { return dimension; }
  inline int32_t get_output_srid() const { return scale; }

  bool operator==(const SQLTypeInfoLite& other) const {
    if (type != other.type)
      return false;
    if (type == ARRAY) {
      if (subtype != other.subtype)
        return false;
      if (subtype == TEXT) {
        if (compression != other.compression)
          return false;
        if (compression == DICT)
          return db_id == other.db_id && dict_id == other.dict_id;
      }
    } else if (type == TEXT) {
      if (compression != other.compression)
        return false;
      if (compression == DICT)
        return db_id == other.db_id && dict_id == other.dict_id;
    } else if (type == POINT || type == LINESTRING || type == POLYGON ||
               type == MULTIPOINT || type == MULTILINESTRING || type == MULTIPOLYGON) {
      if (compression != other.compression)
        return false;
      return get_input_srid() == other.get_input_srid() &&
             get_output_srid() == other.get_output_srid();
    }
    return true;
  }
};

#if !(defined(__CUDACC__) || defined(NO_BOOST))

#include <ostream>

inline std::ostream& operator<<(std::ostream& os, const SQLTypeInfoLite::SQLTypes& type) {
  switch (type) {
    case SQLTypeInfoLite::SQLTypes::UNSPECIFIED:
      os << "UNSPECIFIED";
      break;
    case SQLTypeInfoLite::SQLTypes::BOOLEAN:
      os << "BOOLEAN";
      break;
    case SQLTypeInfoLite::SQLTypes::TINYINT:
      os << "TINYINT";
      break;
    case SQLTypeInfoLite::SQLTypes::SMALLINT:
      os << "SMALLINT";
      break;
    case SQLTypeInfoLite::SQLTypes::INT:
      os << "INT";
      break;
    case SQLTypeInfoLite::SQLTypes::BIGINT:
      os << "BIGINT";
      break;
    case SQLTypeInfoLite::SQLTypes::FLOAT:
      os << "FLOAT";
      break;
    case SQLTypeInfoLite::SQLTypes::DOUBLE:
      os << "DOUBLE";
      break;
    case SQLTypeInfoLite::SQLTypes::POINT:
      os << "POINT";
      break;
    case SQLTypeInfoLite::SQLTypes::LINESTRING:
      os << "LINESTRING";
      break;
    case SQLTypeInfoLite::SQLTypes::POLYGON:
      os << "POLYGON";
      break;
    case SQLTypeInfoLite::SQLTypes::MULTIPOINT:
      os << "MULTIPOINT";
      break;
    case SQLTypeInfoLite::SQLTypes::MULTILINESTRING:
      os << "MULTILINESTRING";
      break;
    case SQLTypeInfoLite::SQLTypes::MULTIPOLYGON:
      os << "MULTIPOLYGON";
      break;
    case SQLTypeInfoLite::SQLTypes::TEXT:
      os << "TEXT";
      break;
    case SQLTypeInfoLite::SQLTypes::ARRAY:
      os << "ARRAY";
      break;
  }
  return os;
}
inline std::ostream& operator<<(std::ostream& os,
                                const SQLTypeInfoLite::EncodingType& type) {
  switch (type) {
    case SQLTypeInfoLite::EncodingType::NONE:
      os << "NONE";
      break;
    case SQLTypeInfoLite::EncodingType::DICT:
      os << "DICT";
      break;
    case SQLTypeInfoLite::EncodingType::GEOINT:
      os << "GEOINT";
      break;
  }
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const SQLTypeInfoLite& ti_lite) {
  os << "SQLTypeInfoLite(";
  os << "type=" << ti_lite.type;
  os << ", subtype=" << ti_lite.subtype;
  os << ", compression=" << ti_lite.compression;
  os << ", dimension=" << ti_lite.dimension;
  os << ", scale=" << ti_lite.scale;
  os << ", db_id=" << ti_lite.db_id;
  os << ", dict_id=" << ti_lite.dict_id;
  os << ")";
  return os;
}
#endif
