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

/*
 * @file    MapDServer.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef MAPDSERVER_H
#define MAPDSERVER_H

#include "gen-cpp/mapd_types.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableGenerations.h"
#include "QueryEngine/TargetMetaInfo.h"

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
    default:
      break;
  }
  abort();
}

inline SQLTypes thrift_to_type(const TDatumType::type& type) {
  switch (type) {
    case TDatumType::BOOL:
      return kBOOLEAN;
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
    default:
      CHECK(false);
  }
  abort();
}

inline std::string thrift_to_name(const TTypeInfo& ti) {
  const auto type = thrift_to_type(ti.type);
  auto internal_ti =
      SQLTypeInfo(ti.is_array ? kARRAY : type, 0, 0, !ti.nullable, kENCODING_NONE, 0, ti.is_array ? type : kNULLT);
  if (type == kDECIMAL || type == kNUMERIC) {
    internal_ti.set_precision(ti.precision);
    internal_ti.set_scale(ti.scale);
  }
  return internal_ti.get_type_name();
}

inline std::string thrift_to_encoding_name(const TTypeInfo& ti) {
  const auto type = thrift_to_type(ti.type);
  const auto encoding = thrift_to_encoding(ti.encoding);
  auto internal_ti =
      SQLTypeInfo(ti.is_array ? kARRAY : type, 0, 0, !ti.nullable, encoding, 0, ti.is_array ? type : kNULLT);
  return internal_ti.get_compression_name();
}

inline SQLTypeInfo type_info_from_thrift(const TTypeInfo& thrift_ti) {
  const auto ti = thrift_to_type(thrift_ti.type);
  const auto base_type = thrift_ti.is_array ? ti : kNULLT;
  const auto type = thrift_ti.is_array ? kARRAY : ti;
  return SQLTypeInfo(type,
                     thrift_ti.precision,
                     thrift_ti.scale,
                     !thrift_ti.nullable,
                     thrift_to_encoding(thrift_ti.encoding),
                     thrift_ti.comp_param,
                     base_type);
}

inline std::vector<TargetMetaInfo> target_meta_infos_from_thrift(const TRowDescriptor& row_desc) {
  std::vector<TargetMetaInfo> target_meta_infos;
  for (const auto& col : row_desc) {
    target_meta_infos.emplace_back(col.col_name, type_info_from_thrift(col.col_type));
  }
  return target_meta_infos;
}

AggregatedColRange column_ranges_from_thrift(const std::vector<TColumnRange>& thrift_column_ranges);

StringDictionaryGenerations string_dictionary_generations_from_thrift(
    const std::vector<TDictionaryGeneration>& thrift_string_dictionary_generations);

TableGenerations table_generations_from_thrift(const std::vector<TTableGeneration>& table_generations);

#endif  // MAPDSERVER_H
