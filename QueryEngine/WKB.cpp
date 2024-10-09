/*
 * Copyright 2024 HEAVY.AI, Inc.
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

#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "Shared/sqltypes.h"

#include "QueryEngine/WKB.h"

extern "C" RUNTIME_EXPORT NEVER_INLINE uint8_t* toWkb(
    size_t* wkb_size,
    int type,  // internal geometry type
    int8_t* coords,
    int64_t coords_size,
    int32_t* meta1,      // e.g. ring_sizes
    int64_t meta1_size,  // e.g. num_rings
    int32_t* meta2,      // e.g. rings (number of rings in each poly)
    int64_t meta2_size,  // e.g. num_polys
    int32_t ic,          // input compression
    int32_t srid_in,     // input srid
    int32_t srid_out,    // output srid
    int32_t* best_planar_srid_ptr) {
  *wkb_size = 0u;
  // decompressed double coords
  auto cv = Geospatial::decompress_coords<double, int32_t>(ic, coords, coords_size);
  auto execute_transform = (srid_in > 0 && srid_out > 0 && srid_in != srid_out);
  if (static_cast<SQLTypes>(type) == kPOINT) {
    Geospatial::GeoPoint point(*cv);
    if (execute_transform && !point.transform(srid_in, srid_out)) {
      return nullptr;
    }
    if (best_planar_srid_ptr) {
      // A non-NULL pointer signifies a request to find the best planar srid
      // to transform this WGS84 geometry to.
      *best_planar_srid_ptr = point.getBestPlanarSRID();
      if (!point.transform(4326, *best_planar_srid_ptr)) {
        return nullptr;
      }
    }
    return point.getWkb(*wkb_size);
  }
  if (static_cast<SQLTypes>(type) == kMULTIPOINT) {
    Geospatial::GeoMultiPoint multipoint(*cv);
    if (execute_transform && !multipoint.transform(srid_in, srid_out)) {
      return nullptr;
    }
    if (best_planar_srid_ptr) {
      // A non-NULL pointer signifies a request to find the best planar srid
      // to transform this WGS84 geometry to, based on geometry's centroid.
      *best_planar_srid_ptr = multipoint.getBestPlanarSRID();
      if (!multipoint.transform(4326, *best_planar_srid_ptr)) {
        return nullptr;
      }
    }
    return multipoint.getWkb(*wkb_size);
  }
  if (static_cast<SQLTypes>(type) == kLINESTRING) {
    Geospatial::GeoLineString linestring(*cv);
    if (execute_transform && !linestring.transform(srid_in, srid_out)) {
      return nullptr;
    }
    if (best_planar_srid_ptr) {
      // A non-NULL pointer signifies a request to find the best planar srid
      // to transform this WGS84 geometry to, based on geometry's centroid.
      *best_planar_srid_ptr = linestring.getBestPlanarSRID();
      if (!linestring.transform(4326, *best_planar_srid_ptr)) {
        return nullptr;
      }
    }
    return linestring.getWkb(*wkb_size);
  }
  std::vector<int32_t> meta1v(meta1, meta1 + meta1_size);
  if (static_cast<SQLTypes>(type) == kMULTILINESTRING) {
    Geospatial::GeoMultiLineString multilinestring(*cv, meta1v);
    if (execute_transform && !multilinestring.transform(srid_in, srid_out)) {
      return nullptr;
    }
    if (best_planar_srid_ptr) {
      // A non-NULL pointer signifies a request to find the best planar srid
      // to transform this WGS84 geometry to, based on geometry's centroid.
      *best_planar_srid_ptr = multilinestring.getBestPlanarSRID();
      if (!multilinestring.transform(4326, *best_planar_srid_ptr)) {
        return nullptr;
      }
    }
    return multilinestring.getWkb(*wkb_size);
  }
  if (static_cast<SQLTypes>(type) == kPOLYGON) {
    Geospatial::GeoPolygon poly(*cv, meta1v);
    if (execute_transform && !poly.transform(srid_in, srid_out)) {
      return nullptr;
    }
    if (best_planar_srid_ptr) {
      // A non-NULL pointer signifies a request to find the best planar srid
      // to transform this WGS84 geometry to, based on geometry's centroid.
      *best_planar_srid_ptr = poly.getBestPlanarSRID();
      if (!poly.transform(4326, *best_planar_srid_ptr)) {
        return nullptr;
      };
    }
    return poly.getWkb(*wkb_size);
  }
  std::vector<int32_t> meta2v(meta2, meta2 + meta2_size);
  if (static_cast<SQLTypes>(type) == kMULTIPOLYGON) {
    // Recognize GEOMETRYCOLLECTION EMPTY encoding
    // MULTIPOLYGON (((0 0,0.00000012345 0.0,0.0 0.00000012345,0 0)))
    // Used to pass along EMPTY from ST_Intersection to ST_IsEmpty for example
    if (meta1_size == 1 && meta2_size == 1) {
      const std::vector<double> ecv = {0.0, 0.0, 0.00000012345, 0.0, 0.0, 0.00000012345};
      if (*cv == ecv) {
        Geospatial::GeoGeometryCollection empty("GEOMETRYCOLLECTION EMPTY");
        return empty.getWkb(*wkb_size);
      }
    }
    Geospatial::GeoMultiPolygon mpoly(*cv, meta1v, meta2v);
    if (execute_transform && !mpoly.transform(srid_in, srid_out)) {
      return nullptr;
    }
    if (best_planar_srid_ptr) {
      // A non-NULL pointer signifies a request to find the best planar srid
      // to transform this WGS84 geometry to, based on geometry's centroid.
      *best_planar_srid_ptr = mpoly.getBestPlanarSRID();
      if (!mpoly.transform(4326, *best_planar_srid_ptr)) {
        return nullptr;
      };
    }
    return mpoly.getWkb(*wkb_size);
  }
  return nullptr;
}

// Conversion form wkb to internal vector representation.
// Each vector components is malloced, caller is reponsible for freeing.
extern "C" RUNTIME_EXPORT NEVER_INLINE bool fromWkb(const uint8_t* wkb_ptr,
                                                    const size_t wkb_size,
                                                    int* result_type,
                                                    int8_t** result_coords,
                                                    int64_t* result_coords_size,
                                                    int32_t** result_meta1,
                                                    int64_t* result_meta1_size,
                                                    int32_t** result_meta2,
                                                    int64_t* result_meta2_size,
                                                    int32_t result_srid_in,
                                                    int32_t result_srid_out,
                                                    int32_t* best_planar_srid_ptr) {
  Geospatial::WkbView wkb_view{wkb_ptr, wkb_size};
  auto result = Geospatial::GeoTypesFactory::createGeoType(wkb_view, false);
  if (!result->isEmpty()) {
    if (best_planar_srid_ptr) {
      // If original geometry has previously been projected to planar srid,
      // need to transform back to WGS84
      if (!result->transform(*best_planar_srid_ptr, 4326)) {
        return false;
      }
    }
    if (result_srid_in > 0 && result_srid_out > 0 and result_srid_in != result_srid_out) {
      if (!result->transform(result_srid_in, result_srid_out)) {
        return false;
      }
    }
  }

  // Get the column values
  std::vector<double> coords{};
  std::vector<int32_t> ring_sizes{};
  std::vector<int32_t> poly_rings{};
  std::vector<double> bounds{};

  // Forcing MULTIPOLYGON result until we can handle any geo.
  if (result->isEmpty()) {
    // Generate a tiny polygon around POINT(0 0), make it a multipolygon
    // MULTIPOLYGON (((0 0,0.00000012345 0.0,0.0 0.00000012345,0 0)))
    // to simulate an empty result
    coords = {0.0, 0.0, 0.00000012345, 0.0, 0.0, 0.00000012345};
    ring_sizes.push_back(3);
    poly_rings.push_back(1);
  } else if (auto result_point = dynamic_cast<Geospatial::GeoPoint*>(result.get())) {
    result_point->getColumns(coords);
    // Generate a tiny polygon around the point, make it a multipolygon
    coords.push_back(coords[0] + 0.0000001);
    coords.push_back(coords[1]);
    coords.push_back(coords[0]);
    coords.push_back(coords[1] + 0.0000001);
    ring_sizes.push_back(3);
    poly_rings.push_back(ring_sizes.size());
  } else if (auto result_poly = dynamic_cast<Geospatial::GeoPolygon*>(result.get())) {
    result_poly->getColumns(coords, ring_sizes, bounds);
    // Convert to a 1-polygon multipolygon
    poly_rings.push_back(ring_sizes.size());
  } else if (auto result_mpoly =
                 dynamic_cast<Geospatial::GeoMultiPolygon*>(result.get())) {
    result_mpoly->getColumns(coords, ring_sizes, poly_rings, bounds);
  } else {
    return false;
  }

  // TODO: consider using a single buffer to hold all components,
  // instead of allocating and registering each component buffer separately

  *result_type = static_cast<int>(kMULTIPOLYGON);

  *result_coords = nullptr;
  int64_t size = coords.size() * sizeof(double);
  if (size > 0) {
    auto* buf = malloc(size);
    if (!buf) {
      return false;
    }
    std::memcpy(buf, coords.data(), size);
    *result_coords = reinterpret_cast<int8_t*>(buf);
  }
  *result_coords_size = size;

  *result_meta1 = nullptr;
  size = ring_sizes.size() * sizeof(int32_t);
  if (size > 0) {
    auto* buf = malloc(size);
    if (!buf) {
      return false;
    }
    std::memcpy(buf, ring_sizes.data(), size);
    *result_meta1 = reinterpret_cast<int32_t*>(buf);
  }
  *result_meta1_size = ring_sizes.size();

  *result_meta2 = nullptr;
  size = poly_rings.size() * sizeof(int32_t);
  if (size > 0) {
    auto* buf = malloc(size);
    if (!buf) {
      return false;
    }
    std::memcpy(buf, poly_rings.data(), size);
    *result_meta2 = reinterpret_cast<int32_t*>(buf);
  }
  *result_meta2_size = poly_rings.size();

  return true;
}
