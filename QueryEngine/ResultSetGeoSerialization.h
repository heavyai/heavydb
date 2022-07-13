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

/**
 * @file    ResultSetGeoSerialization.h
 * @brief   Serialization routines for geospatial types.
 *
 */

#ifndef QUERYENGINE_RESULTSET_GEOSERIALIZATION_H
#define QUERYENGINE_RESULTSET_GEOSERIALIZATION_H

#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/TargetValue.h"
#include "Shared/sqltypes.h"

using VarlenDatumPtr = std::unique_ptr<VarlenDatum>;

using namespace Geospatial;

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoTargetValueSerializer {
  static_assert(IS_GEO(GEO_SOURCE_TYPE), "Invalid geo type for target value serializer.");
};

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoWktSerializer {
  static_assert(IS_GEO(GEO_SOURCE_TYPE), "Invalid geo type for wkt serializer.");
};

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoTargetValuePtrSerializer {
  static_assert(IS_GEO(GEO_SOURCE_TYPE),
                "Invalid geo type for target value ptr serializer.");
};

template <ResultSet::GeoReturnType GEO_RETURN_TYPE, SQLTypes GEO_SOURCE_TYPE>
struct GeoReturnTypeTraits {
  static_assert(GEO_RETURN_TYPE == ResultSet::GeoReturnType::GeoTargetValue ||
                    GEO_RETURN_TYPE == ResultSet::GeoReturnType::WktString ||
                    GEO_RETURN_TYPE == ResultSet::GeoReturnType::GeoTargetValuePtr,
                "ResultSet: Unrecognized Geo Return Type encountered.");
};

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoReturnTypeTraits<ResultSet::GeoReturnType::GeoTargetValue, GEO_SOURCE_TYPE> {
  using GeoSerializerType = GeoTargetValueSerializer<GEO_SOURCE_TYPE>;
};

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoReturnTypeTraits<ResultSet::GeoReturnType::WktString, GEO_SOURCE_TYPE> {
  using GeoSerializerType = GeoWktSerializer<GEO_SOURCE_TYPE>;
};

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoReturnTypeTraits<ResultSet::GeoReturnType::GeoTargetValuePtr, GEO_SOURCE_TYPE> {
  using GeoSerializerType = GeoTargetValuePtrSerializer<GEO_SOURCE_TYPE>;
};

// Point
template <>
struct GeoTargetValueSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (!geo_ti.get_notnull() && vals[0]->is_null) {
      // Alternatively, could decompress vals[0] and check for NULL array sentinel
      return GeoTargetValue(boost::optional<GeoPointTargetValue>{});
    }
    return GeoTargetValue(
        boost::optional<GeoPointTargetValue>{*decompress_coords<double, SQLTypeInfo>(
            geo_ti, vals[0]->pointer, vals[0]->length)});
  }
};

template <>
struct GeoWktSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    // TODO: support EMPTY geo and serialize it as GEOMETRYCOLLECTION EMPTY
    if (!geo_ti.get_notnull() && vals[0]->is_null) {
      return NullableString("NULL");
    }
    Geospatial::GeoPoint point(*decompress_coords<double, SQLTypeInfo>(
        geo_ti, vals[0]->pointer, vals[0]->length));
    return NullableString(point.getWktString());
  }
};

template <>
struct GeoTargetValuePtrSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (!geo_ti.get_notnull() && vals[0]->is_null) {
      // NULL geo
      // Pass along null datum, instead of an empty/null GeoTargetValuePtr
      // return GeoTargetValuePtr();
    }
    return GeoPointTargetValuePtr({std::move(vals[0])});
  }
};

// LineString
template <>
struct GeoTargetValueSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (!geo_ti.get_notnull() && vals[0]->is_null) {
      return GeoTargetValue(boost::optional<GeoLineStringTargetValue>{});
    }
    return GeoTargetValue(
        boost::optional<GeoLineStringTargetValue>{*decompress_coords<double, SQLTypeInfo>(
            geo_ti, vals[0]->pointer, vals[0]->length)});
  }
};

template <>
struct GeoWktSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (!geo_ti.get_notnull() && vals[0]->is_null) {
      // May need to generate "LINESTRING EMPTY" instead of NULL
      return NullableString("NULL");
    }
    Geospatial::GeoLineString linestring(*decompress_coords<double, SQLTypeInfo>(
        geo_ti, vals[0]->pointer, vals[0]->length));
    return NullableString(linestring.getWktString());
  }
};

template <>
struct GeoTargetValuePtrSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (!geo_ti.get_notnull() && vals[0]->is_null) {
      // NULL geo
      // Pass along null datum, instead of an empty/null GeoTargetValuePtr
      // return GeoTargetValuePtr();
    }
    return GeoLineStringTargetValuePtr({std::move(vals[0])});
  }
};

// MultiLineString
template <>
struct GeoTargetValueSerializer<kMULTILINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    if (!geo_ti.get_notnull() && (vals[0]->is_null || vals[1]->is_null)) {
      return GeoTargetValue(boost::optional<GeoMultiLineStringTargetValue>{});
    }
    std::vector<int32_t> linestring_sizes_vec;
    unpack_geo_vector(linestring_sizes_vec, vals[1]->pointer, vals[1]->length);
    auto gtv =
        GeoMultiLineStringTargetValue(*decompress_coords<double, SQLTypeInfo>(
                                          geo_ti, vals[0]->pointer, vals[0]->length),
                                      linestring_sizes_vec);
    return GeoTargetValue(gtv);
  }
};

template <>
struct GeoWktSerializer<kMULTILINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    if (!geo_ti.get_notnull() && (vals[0]->is_null || vals[1]->is_null)) {
      // May need to generate "MULTILINESTRING EMPTY" instead of NULL
      return NullableString("NULL");
    }
    std::vector<int32_t> linestring_sizes_vec;
    unpack_geo_vector(linestring_sizes_vec, vals[1]->pointer, vals[1]->length);
    Geospatial::GeoMultiLineString mls(*decompress_coords<double, SQLTypeInfo>(
                                           geo_ti, vals[0]->pointer, vals[0]->length),
                                       linestring_sizes_vec);
    return NullableString(mls.getWktString());
  };
};

template <>
struct GeoTargetValuePtrSerializer<kMULTILINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    if (!geo_ti.get_notnull() && (vals[0]->is_null || vals[1]->is_null)) {
      // NULL geo
      // Pass along null datum, instead of an empty/null GeoTargetValuePtr
      // return GeoTargetValuePtr();
    }
    return GeoMultiLineStringTargetValuePtr({std::move(vals[0]), std::move(vals[1])});
  }
};

// Polygon
template <>
struct GeoTargetValueSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    if (!geo_ti.get_notnull() && (vals[0]->is_null || vals[1]->is_null)) {
      return GeoTargetValue(boost::optional<GeoPolyTargetValue>{});
    }
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    auto gtv = GeoPolyTargetValue(*decompress_coords<double, SQLTypeInfo>(
                                      geo_ti, vals[0]->pointer, vals[0]->length),
                                  ring_sizes_vec);
    return GeoTargetValue(gtv);
  }
};

template <>
struct GeoWktSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    if (!geo_ti.get_notnull() && (vals[0]->is_null || vals[1]->is_null)) {
      // May need to generate "POLYGON EMPTY" instead of NULL
      return NullableString("NULL");
    }
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    Geospatial::GeoPolygon poly(*decompress_coords<double, SQLTypeInfo>(
                                    geo_ti, vals[0]->pointer, vals[0]->length),
                                ring_sizes_vec);
    return NullableString(poly.getWktString());
  };
};

template <>
struct GeoTargetValuePtrSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    if (!geo_ti.get_notnull() && (vals[0]->is_null || vals[1]->is_null)) {
      // NULL geo
      // Pass along null datum, instead of an empty/null GeoTargetValuePtr
      // return GeoTargetValuePtr();
    }
    return GeoPolyTargetValuePtr({std::move(vals[0]), std::move(vals[1])});
  }
};

// MultiPolygon
template <>
struct GeoTargetValueSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 3>& vals) {
    if (!geo_ti.get_notnull() &&
        (vals[0]->is_null || vals[1]->is_null || vals[2]->is_null)) {
      return GeoTargetValue(boost::optional<GeoMultiPolyTargetValue>{});
    }
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    std::vector<int32_t> poly_rings_vec;
    unpack_geo_vector(poly_rings_vec, vals[2]->pointer, vals[2]->length);
    auto gtv = GeoMultiPolyTargetValue(*decompress_coords<double, SQLTypeInfo>(
                                           geo_ti, vals[0]->pointer, vals[0]->length),
                                       ring_sizes_vec,
                                       poly_rings_vec);
    return GeoTargetValue(gtv);
  }
};

template <>
struct GeoWktSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 3>& vals) {
    if (!geo_ti.get_notnull() &&
        (vals[0]->is_null || vals[1]->is_null || vals[2]->is_null)) {
      // May need to generate "MULTIPOLYGON EMPTY" instead of NULL
      return NullableString("NULL");
    }
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    std::vector<int32_t> poly_rings_vec;
    unpack_geo_vector(poly_rings_vec, vals[2]->pointer, vals[2]->length);
    Geospatial::GeoMultiPolygon mpoly(*decompress_coords<double, SQLTypeInfo>(
                                          geo_ti, vals[0]->pointer, vals[0]->length),
                                      ring_sizes_vec,
                                      poly_rings_vec);
    return NullableString(mpoly.getWktString());
  }
};

template <>
struct GeoTargetValuePtrSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 3>& vals) {
    if (!geo_ti.get_notnull() &&
        (vals[0]->is_null || vals[1]->is_null || vals[2]->is_null)) {
      // NULL geo
      // Pass along null datum, instead of an empty/null GeoTargetValuePtr
      // return GeoTargetValuePtr();
    }
    return GeoMultiPolyTargetValuePtr(
        {std::move(vals[0]), std::move(vals[1]), std::move(vals[2])});
  }
};

#endif  // QUERYENGINE_RESULTSET_GEOSERIALIZATION_H
