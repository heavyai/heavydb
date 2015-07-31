/*
 * @file    MapDServer.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef MAPDSERVER_H
#define	MAPDSERVER_H

#include "gen-cpp/mapd_types.h"
#include "Shared/sqltypes.h"

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
    default:
      break;
  }
  CHECK(false);
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
    default:
      break;
  }
  CHECK(false);
}

#define ENCODING_CASE(encoding)       \
  case kENCODING_##encoding:          \
    return TEncodingType::encoding;

inline TEncodingType::type encoding_to_thrift(const SQLTypeInfo& type_info) {
  switch (type_info.get_compression()) {
  ENCODING_CASE(NONE)
  ENCODING_CASE(FIXED)
  ENCODING_CASE(RL)
  ENCODING_CASE(DIFF)
  ENCODING_CASE(DICT)
  ENCODING_CASE(SPARSE)
  default:
    CHECK(false);
  }
  CHECK(false);
}

#undef ENCODING_CASE

inline std::string thrift_to_name(const TTypeInfo& ti) {
  const auto type = thrift_to_type(ti.type);
  auto internal_ti = SQLTypeInfo(ti.is_array ? kARRAY : type,
    0, 0, !ti.nullable, kENCODING_NONE, 0, ti.is_array ? type : kNULLT);
  return internal_ti.get_type_name();
}

#endif // MAPDSERVER_H
