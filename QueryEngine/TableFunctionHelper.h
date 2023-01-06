/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "ExtensionFunctionsWhitelist.h"

inline ExtArgumentType ext_arg_type_ensure_column(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int8:
      return ExtArgumentType::ColumnInt8;
    case ExtArgumentType::Int16:
      return ExtArgumentType::ColumnInt16;
    case ExtArgumentType::Int32:
      return ExtArgumentType::ColumnInt32;
    case ExtArgumentType::Int64:
      return ExtArgumentType::ColumnInt64;
    case ExtArgumentType::Float:
      return ExtArgumentType::ColumnFloat;
    case ExtArgumentType::Double:
      return ExtArgumentType::ColumnDouble;
    case ExtArgumentType::Bool:
      return ExtArgumentType::ColumnBool;
    case ExtArgumentType::TextEncodingDict:
      return ExtArgumentType::ColumnTextEncodingDict;
    case ExtArgumentType::TextEncodingNone:
      return ExtArgumentType::ColumnTextEncodingNone;
    case ExtArgumentType::ArrayInt8:
      return ExtArgumentType::ColumnArrayInt8;
    case ExtArgumentType::ArrayInt16:
      return ExtArgumentType::ColumnArrayInt16;
    case ExtArgumentType::ArrayInt32:
      return ExtArgumentType::ColumnArrayInt32;
    case ExtArgumentType::ArrayInt64:
      return ExtArgumentType::ColumnArrayInt64;
    case ExtArgumentType::ArrayFloat:
      return ExtArgumentType::ColumnArrayFloat;
    case ExtArgumentType::ArrayDouble:
      return ExtArgumentType::ColumnArrayDouble;
    case ExtArgumentType::ArrayBool:
      return ExtArgumentType::ColumnArrayBool;
    case ExtArgumentType::ArrayTextEncodingDict:
      return ExtArgumentType::ColumnArrayTextEncodingDict;
    case ExtArgumentType::ArrayTextEncodingNone:
      return ExtArgumentType::ColumnArrayTextEncodingNone;
    case ExtArgumentType::GeoPoint:
      return ExtArgumentType::ColumnGeoPoint;
    case ExtArgumentType::GeoLineString:
      return ExtArgumentType::ColumnGeoLineString;
    case ExtArgumentType::GeoPolygon:
      return ExtArgumentType::ColumnGeoPolygon;
    case ExtArgumentType::GeoMultiPoint:
      return ExtArgumentType::ColumnGeoMultiPoint;
    case ExtArgumentType::GeoMultiLineString:
      return ExtArgumentType::ColumnGeoMultiLineString;
    case ExtArgumentType::GeoMultiPolygon:
      return ExtArgumentType::ColumnGeoMultiPolygon;
    default:
      return ext_arg_type;
  }
}

inline ExtArgumentType ext_arg_type_ensure_column_list(
    const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int8:
      return ExtArgumentType::ColumnListInt8;
    case ExtArgumentType::Int16:
      return ExtArgumentType::ColumnListInt16;
    case ExtArgumentType::Int32:
      return ExtArgumentType::ColumnListInt32;
    case ExtArgumentType::Int64:
      return ExtArgumentType::ColumnListInt64;
    case ExtArgumentType::Float:
      return ExtArgumentType::ColumnListFloat;
    case ExtArgumentType::Double:
      return ExtArgumentType::ColumnListDouble;
    case ExtArgumentType::Bool:
      return ExtArgumentType::ColumnListBool;
    case ExtArgumentType::TextEncodingDict:
      return ExtArgumentType::ColumnListTextEncodingDict;
    case ExtArgumentType::TextEncodingNone:
      return ExtArgumentType::ColumnListTextEncodingNone;
    case ExtArgumentType::ArrayInt8:
      return ExtArgumentType::ColumnListArrayInt8;
    case ExtArgumentType::ArrayInt16:
      return ExtArgumentType::ColumnListArrayInt16;
    case ExtArgumentType::ArrayInt32:
      return ExtArgumentType::ColumnListArrayInt32;
    case ExtArgumentType::ArrayInt64:
      return ExtArgumentType::ColumnListArrayInt64;
    case ExtArgumentType::ArrayFloat:
      return ExtArgumentType::ColumnListArrayFloat;
    case ExtArgumentType::ArrayDouble:
      return ExtArgumentType::ColumnListArrayDouble;
    case ExtArgumentType::ArrayBool:
      return ExtArgumentType::ColumnListArrayBool;
    case ExtArgumentType::ArrayTextEncodingDict:
      return ExtArgumentType::ColumnListArrayTextEncodingDict;
    case ExtArgumentType::ArrayTextEncodingNone:
      return ExtArgumentType::ColumnListArrayTextEncodingNone;
    default:
      return ext_arg_type;
  }
}

inline bool is_ext_arg_type_array(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::ArrayInt8:
    case ExtArgumentType::ArrayInt16:
    case ExtArgumentType::ArrayInt32:
    case ExtArgumentType::ArrayInt64:
    case ExtArgumentType::ArrayFloat:
    case ExtArgumentType::ArrayDouble:
    case ExtArgumentType::ArrayBool:
    case ExtArgumentType::ArrayTextEncodingDict:
    case ExtArgumentType::ArrayTextEncodingNone:
      return true;

    default:
      return false;
  }
}

inline bool is_ext_arg_type_column(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::ColumnInt8:
    case ExtArgumentType::ColumnInt16:
    case ExtArgumentType::ColumnInt32:
    case ExtArgumentType::ColumnInt64:
    case ExtArgumentType::ColumnFloat:
    case ExtArgumentType::ColumnDouble:
    case ExtArgumentType::ColumnBool:
    case ExtArgumentType::ColumnTextEncodingDict:
    case ExtArgumentType::ColumnTextEncodingNone:
    case ExtArgumentType::ColumnTimestamp:
    case ExtArgumentType::ColumnArrayInt8:
    case ExtArgumentType::ColumnArrayInt16:
    case ExtArgumentType::ColumnArrayInt32:
    case ExtArgumentType::ColumnArrayInt64:
    case ExtArgumentType::ColumnArrayFloat:
    case ExtArgumentType::ColumnArrayDouble:
    case ExtArgumentType::ColumnArrayBool:
    case ExtArgumentType::ColumnArrayTextEncodingDict:
    case ExtArgumentType::ColumnArrayTextEncodingNone:
    case ExtArgumentType::ColumnGeoPoint:
    case ExtArgumentType::ColumnGeoLineString:
    case ExtArgumentType::ColumnGeoPolygon:
    case ExtArgumentType::ColumnGeoMultiPoint:
    case ExtArgumentType::ColumnGeoMultiLineString:
    case ExtArgumentType::ColumnGeoMultiPolygon:
      return true;

    default:
      return false;
  }
}

inline bool is_ext_arg_type_column_list(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::ColumnListInt8:
    case ExtArgumentType::ColumnListInt16:
    case ExtArgumentType::ColumnListInt32:
    case ExtArgumentType::ColumnListInt64:
    case ExtArgumentType::ColumnListFloat:
    case ExtArgumentType::ColumnListDouble:
    case ExtArgumentType::ColumnListBool:
    case ExtArgumentType::ColumnListTextEncodingDict:
    case ExtArgumentType::ColumnListTextEncodingNone:
    case ExtArgumentType::ColumnListArrayInt8:
    case ExtArgumentType::ColumnListArrayInt16:
    case ExtArgumentType::ColumnListArrayInt32:
    case ExtArgumentType::ColumnListArrayInt64:
    case ExtArgumentType::ColumnListArrayFloat:
    case ExtArgumentType::ColumnListArrayDouble:
    case ExtArgumentType::ColumnListArrayBool:
    case ExtArgumentType::ColumnListArrayTextEncodingDict:
    case ExtArgumentType::ColumnListArrayTextEncodingNone:
    case ExtArgumentType::ColumnListGeoPoint:
    case ExtArgumentType::ColumnListGeoLineString:
    case ExtArgumentType::ColumnListGeoPolygon:
    case ExtArgumentType::ColumnListGeoMultiPoint:
    case ExtArgumentType::ColumnListGeoMultiLineString:
    case ExtArgumentType::ColumnListGeoMultiPolygon:
      return true;

    default:
      return false;
  }
}

inline bool is_ext_arg_type_geo(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::GeoPoint:
    case ExtArgumentType::GeoMultiPoint:
    case ExtArgumentType::GeoLineString:
    case ExtArgumentType::GeoMultiLineString:
    case ExtArgumentType::GeoPolygon:
    case ExtArgumentType::GeoMultiPolygon:
      return true;

    default:
      return false;
  }
}

inline bool is_ext_arg_type_pointer(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
    case ExtArgumentType::PInt16:
    case ExtArgumentType::PInt32:
    case ExtArgumentType::PInt64:
    case ExtArgumentType::PFloat:
    case ExtArgumentType::PDouble:
    case ExtArgumentType::PBool:
      return true;

    default:
      return false;
  }
}

inline bool is_ext_arg_type_scalar(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int8:
    case ExtArgumentType::Int16:
    case ExtArgumentType::Int32:
    case ExtArgumentType::Int64:
    case ExtArgumentType::Float:
    case ExtArgumentType::Double:
    case ExtArgumentType::Bool:
    case ExtArgumentType::TextEncodingDict:
    case ExtArgumentType::TextEncodingNone:
    case ExtArgumentType::Timestamp:
    case ExtArgumentType::DayTimeInterval:
    case ExtArgumentType::YearMonthTimeInterval:
      return true;

    default:
      return false;
  }
}

inline bool is_ext_arg_type_scalar_integer(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int8:
    case ExtArgumentType::Int16:
    case ExtArgumentType::Int32:
    case ExtArgumentType::Int64:
    case ExtArgumentType::Timestamp:
    case ExtArgumentType::DayTimeInterval:
    case ExtArgumentType::YearMonthTimeInterval:
      return true;
    default:
      return false;
  }
}

inline int32_t max_digits_for_ext_integer_arg(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int8:
      return 2;
    case ExtArgumentType::Int16:
      return 4;
    case ExtArgumentType::Int32:
      return 9;
    case ExtArgumentType::Int64:
      return 18;
    default:
      UNREACHABLE();
      return 0;
  }
}

inline bool is_ext_arg_type_nonscalar(const ExtArgumentType ext_arg_type) {
  return !is_ext_arg_type_scalar(ext_arg_type);
}
