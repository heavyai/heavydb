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
  CHECK_EQ(size_t(0), sz % sizeof(T));
  const size_t num_elems = sz / sizeof(T);
  output.resize(num_elems);
  for (size_t i = 0; i < num_elems; i++) {
    output[i] = elems[i];
  }
}

template <class T>
T wrap_decompressed_coord(const double val) {
  return static_cast<T>(val);
}

template <>
double wrap_decompressed_coord(const double val) {
  return val;
}

// TODO(adb): Move this to a common geo compression file / class
template <typename T>
void decompress_geo_coords_geoint32(std::vector<T>& dec,
                                    const int8_t* enc,
                                    const size_t sz) {
  const auto compressed_coords = reinterpret_cast<const int32_t*>(enc);
  bool x = true;
  dec.reserve(sz / sizeof(int32_t));
  for (size_t i = 0; i < sz / sizeof(int32_t); i++) {
    // decompress longitude: -2,147,483,647..2,147,483,647  --->  -180..180
    // decompress latitude:  -2,147,483,647..2,147,483,647  --->   -90..90
    double decompressed_coord =
        (x ? 180.0 : 90.0) * (compressed_coords[i] / 2147483647.0);
    dec.push_back(wrap_decompressed_coord<T>(decompressed_coord));
    x = !x;
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

// Point
template <>
struct GeoTargetValueSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz) {
    return GeoPointTargetValue(*decompress_coords<double>(geo_ti, coords, coords_sz));
  }
};

template <>
struct GeoWktSerializer<kPOINT> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz) {
    Geo_namespace::GeoPoint point(*decompress_coords<double>(geo_ti, coords, coords_sz));
    return NullableString(point.getWktString());
  }
};

// LineString
template <>
struct GeoTargetValueSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz) {
    return GeoLineStringTargetValue(
        *decompress_coords<double>(geo_ti, coords, coords_sz));
  }
};

template <>
struct GeoWktSerializer<kLINESTRING> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz) {
    Geo_namespace::GeoLineString linestring(
        *decompress_coords<double>(geo_ti, coords, coords_sz));
    return NullableString(linestring.getWktString());
  }
};

// Polygon
template <>
struct GeoTargetValueSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz,
                                      const int8_t* ring_sizes,
                                      const size_t ring_sizes_sz) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, ring_sizes, ring_sizes_sz);
    return GeoPolyTargetValue(*decompress_coords<double>(geo_ti, coords, coords_sz),
                              ring_sizes_vec);
  }
};

template <>
struct GeoWktSerializer<kPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz,
                                      const int8_t* ring_sizes,
                                      const size_t ring_sizes_sz) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, ring_sizes, ring_sizes_sz);
    Geo_namespace::GeoPolygon poly(*decompress_coords<double>(geo_ti, coords, coords_sz),
                                   ring_sizes_vec);
    return NullableString(poly.getWktString());
  };
};

// MultiPolygon
template <>
struct GeoTargetValueSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz,
                                      const int8_t* ring_sizes,
                                      const size_t ring_sizes_sz,
                                      const int8_t* poly_rings,
                                      const size_t poly_rings_sz) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, ring_sizes, ring_sizes_sz);
    std::vector<int32_t> poly_rings_vec;
    unpack_geo_vector(poly_rings_vec, poly_rings, poly_rings_sz);
    return GeoMultiPolyTargetValue(*decompress_coords<double>(geo_ti, coords, coords_sz),
                                   ring_sizes_vec,
                                   poly_rings_vec);
  }
};

template <>
struct GeoWktSerializer<kMULTIPOLYGON> {
  static inline TargetValue serialize(const SQLTypeInfo& geo_ti,
                                      const int8_t* coords,
                                      const size_t coords_sz,
                                      const int8_t* ring_sizes,
                                      const size_t ring_sizes_sz,
                                      const int8_t* poly_rings,
                                      const size_t poly_rings_sz) {
    std::vector<int32_t> ring_sizes_vec;
    unpack_geo_vector(ring_sizes_vec, ring_sizes, ring_sizes_sz);
    std::vector<int32_t> poly_rings_vec;
    unpack_geo_vector(poly_rings_vec, poly_rings, poly_rings_sz);
    Geo_namespace::GeoMultiPolygon mpoly(
        *decompress_coords<double>(geo_ti, coords, coords_sz),
        ring_sizes_vec,
        poly_rings_vec);
    return NullableString(mpoly.getWktString());
  }
};

#endif  // QUERYENGINE_RESULTSET_GEOSERIALIZATION_H