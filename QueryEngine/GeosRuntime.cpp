/*
 * Copyright 2019 OmniSci Technologies, Inc.
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

#ifdef ENABLE_GEOS

#ifndef __CUDACC__

#include <cstdarg>
#include <mutex>

#include "../Shared/checked_alloc.h"
#include "../Shared/funcannotations.h"
#include "../Shared/geo_compression.h"
#include "../Shared/geo_types.h"
#include "GeosRuntime.h"

#define GEOS_USE_ONLY_R_API
#include <geos_c.h>

using namespace Geo_namespace;

using WKB = std::vector<uint8_t>;

#define MAX_GEOS_MESSAGE_LEN 200

static std::mutex geos_log_info_mutex;
static std::mutex geos_log_error_mutex;

// called by GEOS on notice
static void geos_notice_handler(const char* fmt, ...) {
  char buffer[MAX_GEOS_MESSAGE_LEN];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, MAX_GEOS_MESSAGE_LEN, fmt, args);
  va_end(args);
  {
    std::lock_guard<std::mutex> guard(geos_log_info_mutex);
    LOG(INFO) << "GEOS Notice: " << std::string(buffer);
  }
}

// called by GEOS on error
static void geos_error_handler(const char* fmt, ...) {
  va_list args;
  char buffer[MAX_GEOS_MESSAGE_LEN];
  va_start(args, fmt);
  vsnprintf(buffer, MAX_GEOS_MESSAGE_LEN, fmt, args);
  va_end(args);
  {
    std::lock_guard<std::mutex> guard(geos_log_error_mutex);
    LOG(ERROR) << "GEOS Error: " << std::string(buffer);
  }
}

GEOSContextHandle_t create_context() {
#if GEOS_VERSION_MAJOR == 3 && GEOS_VERSION_MINOR < 5
  GEOSContextHandle_t context = initGEOS_r(geos_notice_handler, geos_error_handler);
  CHECK(context);
  return context;
#else
  GEOSContextHandle_t context = GEOS_init_r();
  CHECK(context);
  GEOSContext_setNoticeHandler_r(context, geos_notice_handler);
  GEOSContext_setErrorHandler_r(context, geos_error_handler);
  return context;
#endif
}

void destroy_context(GEOSContextHandle_t context) {
  CHECK(context);
#if GEOS_VERSION_MAJOR == 3 && GEOS_VERSION_MINOR < 5
  finishGEOS_r(context);
#else
  GEOS_finish_r(context);
#endif
}

bool toWkb(WKB& wkb,
           int type,  // internal geometry type
           int8_t* coords,
           int64_t coords_size,
           int32_t* meta1,      // e.g. ring_sizes
           int64_t meta1_size,  // e.g. num_rings
           int32_t* meta2,      // e.g. rings (number of rings in each poly)
           int64_t meta2_size,  // e.g. num_polys
           int32_t ic           // input compression
) {
  // decompressed double coords
  auto cv = geospatial::decompress_coords<double, int32_t>(ic, coords, coords_size);
  if (static_cast<SQLTypes>(type) == kPOINT) {
    GeoPoint point(*cv);
    return point.getWkb(wkb);
  }
  if (static_cast<SQLTypes>(type) == kLINESTRING) {
    GeoLineString linestring(*cv);
    return linestring.getWkb(wkb);
  }
  std::vector<int32_t> meta1v(meta1, meta1 + meta1_size);
  if (static_cast<SQLTypes>(type) == kPOLYGON) {
    GeoPolygon poly(*cv, meta1v);
    return poly.getWkb(wkb);
  }
  std::vector<int32_t> meta2v(meta2, meta2 + meta2_size);
  if (static_cast<SQLTypes>(type) == kMULTIPOLYGON) {
    // Recognize GEOMETRYCOLLECTION EMPTY encoding
    // MULTIPOLYGON (((0 0,0.00000012345 0.0,0.0 0.00000012345,0 0)))
    // Used to pass along EMPTY from ST_Intersection to ST_IsEmpty for example
    if (meta1_size == 1 && meta2_size == 1) {
      const std::vector<double> ecv = {0.0, 0.0, 0.00000012345, 0.0, 0.0, 0.00000012345};
      if (*cv == ecv) {
        GeoGeometryCollection empty("GEOMETRYCOLLECTION EMPTY");
        return empty.getWkb(wkb);
      }
    }
    GeoMultiPolygon mpoly(*cv, meta1v, meta2v);
    return mpoly.getWkb(wkb);
  }
  return false;
}

// Conversion form wkb to internal vector representation.
// Each vector components is malloced, caller is reponsible for freeing.
bool fromWkb(WKB& wkb,
             int* result_type,
             int8_t** result_coords,
             int64_t* result_coords_size,
             int32_t** result_meta1,
             int64_t* result_meta1_size,
             int32_t** result_meta2,
             int64_t* result_meta2_size) {
  auto result = GeoTypesFactory::createGeoType(wkb);

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
  } else if (auto result_point = dynamic_cast<GeoPoint*>(result.get())) {
    result_point->getColumns(coords);
    // Generate a tiny polygon around the point, make it a multipolygon
    coords.push_back(coords[0] + 0.0000001);
    coords.push_back(coords[1]);
    coords.push_back(coords[0]);
    coords.push_back(coords[1] + 0.0000001);
    ring_sizes.push_back(3);
    poly_rings.push_back(ring_sizes.size());
  } else if (auto result_poly = dynamic_cast<GeoPolygon*>(result.get())) {
    result_poly->getColumns(coords, ring_sizes, bounds);
    // Convert to a 1-polygon multipolygon
    poly_rings.push_back(ring_sizes.size());
  } else if (auto result_mpoly = dynamic_cast<GeoMultiPolygon*>(result.get())) {
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
    auto buf = checked_malloc(size);
    std::memcpy(buf, coords.data(), size);
    *result_coords = reinterpret_cast<int8_t*>(buf);
  }
  *result_coords_size = size;

  *result_meta1 = nullptr;
  size = ring_sizes.size() * sizeof(int32_t);
  if (size > 0) {
    auto buf = checked_malloc(size);
    std::memcpy(buf, ring_sizes.data(), size);
    *result_meta1 = reinterpret_cast<int32_t*>(buf);
  }
  *result_meta1_size = ring_sizes.size();

  *result_meta2 = nullptr;
  size = poly_rings.size() * sizeof(int32_t);
  if (size > 0) {
    auto buf = checked_malloc(size);
    std::memcpy(buf, poly_rings.data(), size);
    *result_meta2 = reinterpret_cast<int32_t*>(buf);
  }
  *result_meta2_size = poly_rings.size();

  return true;
}
#endif

extern "C" bool Geos_Wkb_Wkb(int op,
                             int arg1_type,
                             int8_t* arg1_coords,
                             int64_t arg1_coords_size,
                             int32_t* arg1_meta1,
                             int64_t arg1_meta1_size,
                             int32_t* arg1_meta2,
                             int64_t arg1_meta2_size,
                             // TODO: add meta3 args to support generic geometries
                             int32_t arg1_ic,
                             int32_t arg1_srid,
                             int arg2_type,
                             int8_t* arg2_coords,
                             int64_t arg2_coords_size,
                             int32_t* arg2_meta1,
                             int64_t arg2_meta1_size,
                             int32_t* arg2_meta2,
                             int64_t arg2_meta2_size,
                             // TODO: add meta3 args to support generic geometries
                             int32_t arg2_ic,
                             int32_t arg2_srid,
                             // TODO: add transform args
                             int* result_type,
                             int8_t** result_coords,
                             int64_t* result_coords_size,
                             int32_t** result_meta1,
                             int64_t* result_meta1_size,
                             int32_t** result_meta2,
                             int64_t* result_meta2_size) {
#ifndef __CUDACC__
  // Get the result geo
  // What if intersection is not a POLYGON? POINT? LINESTRING, MULTIPOLYGON?
  // What if intersection is empty? Return null buffer pointers? Return false?
  // What if geos fails?

  WKB wkb1{};
  if (!toWkb(wkb1,
             arg1_type,
             arg1_coords,
             arg1_coords_size,
             arg1_meta1,
             arg1_meta1_size,
             arg1_meta2,
             arg1_meta2_size,
             arg1_ic)) {
    return false;
  }
  WKB wkb2{};
  if (!toWkb(wkb2,
             arg2_type,
             arg2_coords,
             arg2_coords_size,
             arg2_meta1,
             arg2_meta1_size,
             arg2_meta2,
             arg2_meta2_size,
             arg2_ic)) {
    return false;
  }
  auto status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb1.data(), wkb1.size());
  if (g1) {
    GEOSSetSRID_r(context, g1, arg1_srid);
    auto* g2 = GEOSGeomFromWKB_buf_r(context, wkb2.data(), wkb2.size());
    if (g2) {
      GEOSSetSRID_r(context, g2, arg2_srid);
      GEOSGeometry* g = nullptr;
      if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kINTERSECTION) {
        g = GEOSIntersection_r(context, g1, g2);
      } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kDIFFERENCE) {
        g = GEOSDifference_r(context, g1, g2);
      } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kUNION) {
        g = GEOSUnion_r(context, g1, g2);
      }
      if (g) {
        size_t wkb_size = 0ULL;
        auto wkb_buf = GEOSGeomToWKB_buf_r(context, g, &wkb_size);
        if (wkb_buf && wkb_size > 0ULL) {
          WKB wkb(wkb_buf, wkb_buf + wkb_size);
          free(wkb_buf);
          status = fromWkb(wkb,
                           result_type,
                           result_coords,
                           result_coords_size,
                           result_meta1,
                           result_meta1_size,
                           result_meta2,
                           result_meta2_size);
        }
        GEOSGeom_destroy_r(context, g);
      }
      GEOSGeom_destroy_r(context, g2);
    }
    GEOSGeom_destroy_r(context, g1);
  }
  destroy_context(context);
  return status;
#else
  return false;
#endif
}

extern "C" bool Geos_Wkb_double(int op,
                                int arg1_type,
                                int8_t* arg1_coords,
                                int64_t arg1_coords_size,
                                int32_t* arg1_meta1,
                                int64_t arg1_meta1_size,
                                int32_t* arg1_meta2,
                                int64_t arg1_meta2_size,
                                // TODO: add meta3 args to support generic geometries
                                int32_t arg1_ic,
                                int32_t arg1_srid,
                                double arg2,
                                // TODO: add transform args
                                int* result_type,
                                int8_t** result_coords,
                                int64_t* result_coords_size,
                                int32_t** result_meta1,
                                int64_t* result_meta1_size,
                                int32_t** result_meta2,
                                int64_t* result_meta2_size) {
#ifndef __CUDACC__
  WKB wkb1{};
  if (!toWkb(wkb1,
             arg1_type,
             arg1_coords,
             arg1_coords_size,
             arg1_meta1,
             arg1_meta1_size,
             arg1_meta2,
             arg1_meta2_size,
             arg1_ic)) {
    return false;
  }

  auto status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb1.data(), wkb1.size());
  if (g1) {
    GEOSSetSRID_r(context, g1, arg1_srid);
    GEOSGeometry* g = nullptr;
    if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kBUFFER) {
      int quadsegs = 8;  // default
      g = GEOSBuffer_r(context, g1, arg2, quadsegs);
    }
    if (g) {
      size_t wkb_size = 0ULL;
      auto wkb_buf = GEOSGeomToWKB_buf_r(context, g, &wkb_size);
      if (wkb_buf && wkb_size > 0ULL) {
        WKB wkb(wkb_buf, wkb_buf + wkb_size);
        free(wkb_buf);
        status = fromWkb(wkb,
                         result_type,
                         result_coords,
                         result_coords_size,
                         result_meta1,
                         result_meta1_size,
                         result_meta2,
                         result_meta2_size);
      }
      GEOSGeom_destroy_r(context, g);
    }
    GEOSGeom_destroy_r(context, g1);
  }
  destroy_context(context);
  return status;
#else
  return false;
#endif
}

extern "C" bool Geos_Wkb(int op,
                         int arg_type,
                         int8_t* arg_coords,
                         int64_t arg_coords_size,
                         int32_t* arg_meta1,
                         int64_t arg_meta1_size,
                         int32_t* arg_meta2,
                         int64_t arg_meta2_size,
                         // TODO: add meta3 args to support generic geometries
                         int32_t arg_ic,
                         int32_t arg_srid,
                         bool* result) {
#ifndef __CUDACC__
  WKB wkb1{};
  if (!result || !toWkb(wkb1,
                        arg_type,
                        arg_coords,
                        arg_coords_size,
                        arg_meta1,
                        arg_meta1_size,
                        arg_meta2,
                        arg_meta2_size,
                        arg_ic)) {
    return false;
  }

  auto status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb1.data(), wkb1.size());
  if (g1) {
    GEOSSetSRID_r(context, g1, arg_srid);
    if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kISEMPTY) {
      *result = GEOSisEmpty_r(context, g1);
      status = true;
    } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kISVALID) {
      *result = GEOSisValid_r(context, g1);
      status = true;
    }
    GEOSGeom_destroy_r(context, g1);
  }
  destroy_context(context);
  return status;
#else
  return false;
#endif
}

#endif
