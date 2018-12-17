/*
 * Copyright 2018 MapD Technologies, Inc.
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
 * @author  Alex Baden <alex.baden@mapd.com>
 * @brief   Serialization routines for geospatial types.
 *
 */

#ifndef QUERYENGINE_RESULTSET_GEOSERIALIZATION_H
#define QUERYENGINE_RESULTSET_GEOSERIALIZATION_H

#include <Shared/geo_compression.h>
#include "TargetValue.h"

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoTargetValueSerializer {
  static_assert(IS_GEO(GEO_SOURCE_TYPE), "Invalid geo type for target value serializer.");
};

template <SQLTypes GEO_SOURCE_TYPE>
struct GeoWktSerializer {
  static_assert(IS_GEO(GEO_SOURCE_TYPE), "Invalid geo type for wkt serializer.");
};

template <ResultSet::GeoReturnType GEO_RETURN_TYPE, SQLTypes GEO_SOURCE_TYPE>
struct GeoReturnTypeTraits {
  static_assert(GEO_RETURN_TYPE == ResultSet::GeoReturnType::GeoTargetValue ||
                    GEO_RETURN_TYPE == ResultSet::GeoReturnType::WktString,
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

namespace {

template <typename T>
void unpack_geo_vector(std::vector<T>& output, const int8_t* input_ptr, const size_t sz) {
  auto elems = reinterpret_cast<const T*>(input_ptr);
  CHECK(elems);
  CHECK_EQ(size_t(0), sz % sizeof(T));
  const size_t num_elems = sz / sizeof(T);
  output.resize(num_elems);
  for (size_t i = 0; i < num_elems; i++) {
    output[i] = elems[i];
  }
}

template <typename T>
void decompress_geo_coords_geoint32(std::vector<T>& dec,
                                    const int8_t* enc,
                                    const size_t sz) {
  const auto compressed_coords = reinterpret_cast<const int32_t*>(enc);
  dec.reserve(sz / sizeof(int32_t));
  for (size_t i = 0; i < sz / sizeof(int32_t); i += 2) {
    dec.push_back(
        Geo_namespace::decompress_longitude_coord_geoint32(compressed_coords[i]));
    dec.push_back(
        Geo_namespace::decompress_lattitude_coord_geoint32(compressed_coords[i + 1]));
  }
}

template <typename T>
std::shared_ptr<std::vector<T>> decompress_coords(const SQLTypeInfo& geo_ti,
                                                  const int8_t* coords,
                                                  const size_t coords_sz);

template <>
std::shared_ptr<std::vector<double>> decompress_coords<double>(const SQLTypeInfo& geo_ti,
                                                               const int8_t* coords,
                                                               const size_t coords_sz) {
  auto decompressed_coords_ptr = std::make_shared<std::vector<double>>();
  if (geo_ti.get_compression() == kENCODING_GEOINT) {
    if (geo_ti.get_comp_param() == 32) {
      decompress_geo_coords_geoint32(*decompressed_coords_ptr, coords, coords_sz);
    }
  } else {
    CHECK_EQ(geo_ti.get_compression(), kENCODING_NONE);
    unpack_geo_vector(*decompressed_coords_ptr, coords, coords_sz);
  }
  return decompressed_coords_ptr;
}

}  // namespace

using VarlenDatumPtr = std::unique_ptr<VarlenDatum>;

// Point
template <>
struct GeoTargetValueSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (vals[0]->is_null) {
      return GeoPointTargetValue(std::vector<double>{});
    }
    return GeoPointTargetValue(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length));
  }
};

template <>
struct GeoWktSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    Geo_namespace::GeoPoint point(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length));
    return NullableString(point.getWktString());
  }
};

// LineString
template <>
struct GeoTargetValueSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    if (vals[0]->is_null) {
      return GeoLineStringTargetValue(std::vector<double>{});
    }
    return GeoLineStringTargetValue(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length));
  }
};

template <>
struct GeoWktSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 1>& vals) {
    Geo_namespace::GeoLineString linestring(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length));
    return NullableString(linestring.getWktString());
  }
};

// Polygon
template <>
struct GeoTargetValueSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    return GeoPolyTargetValue(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length),
        ring_sizes_vec);
  }
};

template <>
struct GeoWktSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 2>& vals) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    Geo_namespace::GeoPolygon poly(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length),
        ring_sizes_vec);
    return NullableString(poly.getWktString());
  };
};

// MultiPolygon
template <>
struct GeoTargetValueSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 3>& vals) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    std::vector<int32_t> poly_rings_vec;
    unpack_geo_vector(poly_rings_vec, vals[2]->pointer, vals[2]->length);
    return GeoMultiPolyTargetValue(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length),
        ring_sizes_vec,
        poly_rings_vec);
  }
};

template <>
struct GeoWktSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      std::array<VarlenDatumPtr, 3>& vals) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, vals[1]->pointer, vals[1]->length);
    std::vector<int32_t> poly_rings_vec;
    unpack_geo_vector(poly_rings_vec, vals[2]->pointer, vals[2]->length);
    Geo_namespace::GeoMultiPolygon mpoly(
        *decompress_coords<double>(geo_ti, vals[0]->pointer, vals[0]->length),
        ring_sizes_vec,
        poly_rings_vec);
    return NullableString(mpoly.getWktString());
  }
};

#endif  // QUERYENGINE_RESULTSET_GEOSERIALIZATION_H
