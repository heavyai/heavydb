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

#ifndef THRIFT_TYPE_CONVERT_H
#define THRIFT_TYPE_CONVERT_H

#include "gen-cpp/mapd_types.h"

#include <glog/logging.h>

#include <string>

inline TDatumType::type type_to_thrift(const SQLTypeInfo& type_info) {
  SQLTypes type = type_info.get_type();
  if (type == kARRAY) {
    type = type_info.get_subtype();
  }
  switch (type) {
    case kBOOLEAN:
      return TDatumType::BOOL;
    case kTINYINT:
      return TDatumType::TINYINT;
    case kSMALLINT:
      return TDatumType::SMALLINT;
    case kINT:
      return TDatumType::INT;
    case kBIGINT:
      return TDatumType::BIGINT;
    case kFLOAT:
      return TDatumType::FLOAT;
    case kNUMERIC:
    case kDECIMAL:
      return TDatumType::DECIMAL;
    case kDOUBLE:
      return TDatumType::DOUBLE;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return TDatumType::STR;
    case kTIME:
      return TDatumType::TIME;
    case kTIMESTAMP:
      return TDatumType::TIMESTAMP;
    case kDATE:
      return TDatumType::DATE;
    case kINTERVAL_DAY_TIME:
      return TDatumType::INTERVAL_DAY_TIME;
    case kINTERVAL_YEAR_MONTH:
      return TDatumType::INTERVAL_YEAR_MONTH;
    case kPOINT:
      return TDatumType::POINT;
    case kLINESTRING:
      return TDatumType::LINESTRING;
    case kPOLYGON:
      return TDatumType::POLYGON;
    case kMULTIPOLYGON:
      return TDatumType::MULTIPOLYGON;
    case kGEOMETRY:
      return TDatumType::GEOMETRY;
    case kGEOGRAPHY:
      return TDatumType::GEOGRAPHY;
    default:
      break;
  }
  abort();
}

inline SQLTypes thrift_to_type(const TDatumType::type& type) {
  switch (type) {
    case TDatumType::BOOL:
      return kBOOLEAN;
    case TDatumType::TINYINT:
      return kTINYINT;
    case TDatumType::SMALLINT:
      return kSMALLINT;
    case TDatumType::INT:
      return kINT;
    case TDatumType::BIGINT:
      return kBIGINT;
    case TDatumType::FLOAT:
      return kFLOAT;
    case TDatumType::DECIMAL:
      return kDECIMAL;
    case TDatumType::DOUBLE:
      return kDOUBLE;
    case TDatumType::STR:
      return kTEXT;
    case TDatumType::TIME:
      return kTIME;
    case TDatumType::TIMESTAMP:
      return kTIMESTAMP;
    case TDatumType::DATE:
      return kDATE;
    case TDatumType::INTERVAL_DAY_TIME:
      return kINTERVAL_DAY_TIME;
    case TDatumType::INTERVAL_YEAR_MONTH:
      return kINTERVAL_YEAR_MONTH;
    case TDatumType::POINT:
      return kPOINT;
    case TDatumType::LINESTRING:
      return kLINESTRING;
    case TDatumType::POLYGON:
      return kPOLYGON;
    case TDatumType::MULTIPOLYGON:
      return kMULTIPOLYGON;
    case TDatumType::GEOMETRY:
      return kGEOMETRY;
    case TDatumType::GEOGRAPHY:
      return kGEOGRAPHY;
    default:
      break;
  }
  abort();
}

#define THRIFT_ENCODING_CASE(encoding) \
  case kENCODING_##encoding:           \
    return TEncodingType::encoding;

#define UNTHRIFT_ENCODING_CASE(encoding) \
  case TEncodingType::encoding:          \
    return kENCODING_##encoding;

inline TEncodingType::type encoding_to_thrift(const SQLTypeInfo& type_info) {
  switch (type_info.get_compression()) {
    THRIFT_ENCODING_CASE(NONE)
    THRIFT_ENCODING_CASE(FIXED)
    THRIFT_ENCODING_CASE(RL)
    THRIFT_ENCODING_CASE(DIFF)
    THRIFT_ENCODING_CASE(DICT)
    THRIFT_ENCODING_CASE(SPARSE)
    THRIFT_ENCODING_CASE(GEOINT)
    THRIFT_ENCODING_CASE(DATE_IN_DAYS)
    default:
      CHECK(false);
  }
  abort();
}

#undef ENCODING_CASE

inline EncodingType thrift_to_encoding(const TEncodingType::type tEncodingType) {
  switch (tEncodingType) {
    UNTHRIFT_ENCODING_CASE(NONE)
    UNTHRIFT_ENCODING_CASE(FIXED)
    UNTHRIFT_ENCODING_CASE(RL)
    UNTHRIFT_ENCODING_CASE(DIFF)
    UNTHRIFT_ENCODING_CASE(DICT)
    UNTHRIFT_ENCODING_CASE(SPARSE)
    UNTHRIFT_ENCODING_CASE(GEOINT)
    UNTHRIFT_ENCODING_CASE(DATE_IN_DAYS)
    default:
      CHECK(false);
  }
  abort();
}

inline std::string thrift_to_name(const TTypeInfo& ti) {
  const auto type = thrift_to_type(ti.type);
  auto internal_ti = SQLTypeInfo(ti.is_array ? kARRAY : type,
                                 0,
                                 0,
                                 !ti.nullable,
                                 kENCODING_NONE,
                                 0,
                                 ti.is_array ? type : kNULLT);
  if (type == kDECIMAL || type == kNUMERIC) {
    internal_ti.set_precision(ti.precision);
    internal_ti.set_scale(ti.scale);
  } else if (type == kTIMESTAMP) {
    internal_ti.set_precision(ti.precision);
  }
  if (IS_GEO(type)) {
    internal_ti.set_subtype(static_cast<SQLTypes>(ti.precision));
    internal_ti.set_input_srid(ti.scale);
    internal_ti.set_output_srid(ti.scale);
  }
  internal_ti.set_size(ti.size);
  return internal_ti.get_type_name();
}

inline std::string thrift_to_encoding_name(const TTypeInfo& ti) {
  const auto type = thrift_to_type(ti.type);
  const auto encoding = thrift_to_encoding(ti.encoding);
  auto internal_ti = SQLTypeInfo(ti.is_array ? kARRAY : type,
                                 0,
                                 0,
                                 !ti.nullable,
                                 encoding,
                                 0,
                                 ti.is_array ? type : kNULLT);
  return internal_ti.get_compression_name();
}

inline SQLTypeInfo type_info_from_thrift(const TTypeInfo& thrift_ti,
                                         const bool strip_geo_encoding = false) {
  const auto ti = thrift_to_type(thrift_ti.type);
  if (IS_GEO(ti)) {
    const auto base_type = static_cast<SQLTypes>(thrift_ti.precision);
    return SQLTypeInfo(
        ti,
        thrift_ti.scale,
        thrift_ti.scale,
        !thrift_ti.nullable,
        strip_geo_encoding ? kENCODING_NONE : thrift_to_encoding(thrift_ti.encoding),
        thrift_ti.comp_param,
        base_type);
  }
  if (thrift_ti.is_array) {
    auto ati = SQLTypeInfo(kARRAY,
                           thrift_ti.precision,
                           thrift_ti.scale,
                           !thrift_ti.nullable,
                           thrift_to_encoding(thrift_ti.encoding),
                           thrift_ti.comp_param,
                           ti);
    ati.set_size(thrift_ti.size);
    return ati;
  }
  return SQLTypeInfo(ti,
                     thrift_ti.precision,
                     thrift_ti.scale,
                     !thrift_ti.nullable,
                     thrift_to_encoding(thrift_ti.encoding),
                     thrift_ti.comp_param,
                     kNULLT);
}

#endif  // THRIFT_TYPE_CONVERT_H
