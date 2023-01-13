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

#include <memory>

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
  int32_t dimension;         // input_srid
  int32_t scale;             // output_srid
  int32_t db_id;             // used by TEXT and ARRAY of TEXT
  int32_t dict_id;           // used by TEXT and ARRAY of TEXT

  inline bool is_geoint() const { return compression == GEOINT; }
  inline int32_t get_input_srid() const { return dimension; }
  inline int32_t get_output_srid() const { return scale; }
};
