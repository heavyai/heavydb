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

#ifdef ENABLE_GEOS

#ifndef __CUDACC__

#include <cstdarg>
#include <cstring>
#include <mutex>

#ifdef NO_BOOST
#define SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS
#endif

#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "QueryEngine/GeosRuntime.h"
#include "QueryEngine/WKB.h"
#include "Shared/funcannotations.h"

#define GEOS_USE_ONLY_R_API
#include <geos_c.h>

using namespace Geospatial;

#define MAX_GEOS_MESSAGE_LEN 200

static std::mutex geos_log_info_mutex;
static std::mutex geos_log_error_mutex;

namespace {
struct FreeDeleter {
  void operator()(uint8_t* p) { free(p); }
};
using WkbUniquePtr = std::unique_ptr<uint8_t, FreeDeleter>;
}  // namespace

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

GEOSGeometry* postprocess(GEOSContextHandle_t context, GEOSGeometry* g) {
  if (g && GEOSisEmpty_r(context, g) == 0) {
    auto type = GEOSGeomTypeId_r(context, g);
    if (type != -1) {
      if (type != GEOS_POINT && type != GEOS_POLYGON && type != GEOS_MULTIPOLYGON) {
        int quadsegs = 1;  // coarse
        double tiny_distance = 0.000000001;
        auto ng = GEOSBuffer_r(context, g, tiny_distance, quadsegs);
        GEOSGeom_destroy_r(context, g);
        return ng;
      }
    }
  }
  return g;
}
#endif

extern "C" RUNTIME_EXPORT bool Geos_Wkb_Wkb(
    int32_t op,
    int32_t arg1_type,
    int8_t* arg1_coords,
    int64_t arg1_coords_size,
    int32_t* arg1_meta1,
    int64_t arg1_meta1_size,
    int32_t* arg1_meta2,
    int64_t arg1_meta2_size,
    // TODO: add meta3 args to support generic geometries
    int32_t arg1_ic,
    int32_t arg1_srid_in,
    int32_t arg1_srid_out,
    int32_t arg2_type,
    int8_t* arg2_coords,
    int64_t arg2_coords_size,
    int32_t* arg2_meta1,
    int64_t arg2_meta1_size,
    int32_t* arg2_meta2,
    int64_t arg2_meta2_size,
    // TODO: add meta3 args to support generic geometries
    int32_t arg2_ic,
    int32_t arg2_srid_in,
    int32_t arg2_srid_out,
    // TODO: add transform args
    int32_t* result_type,
    int8_t** result_coords,
    int64_t* result_coords_size,
    int32_t** result_meta1,
    int64_t* result_meta1_size,
    int32_t** result_meta2,
    int64_t* result_meta2_size,
    // TODO: add support for output compression
    int32_t result_srid_out) {
#ifndef __CUDACC__
  // Get the result geo
  // What if intersection is not a POLYGON? POINT? LINESTRING, MULTIPOLYGON?
  // What if intersection is empty? Return null buffer pointers? Return false?
  // What if geos fails?

  int32_t* best_planar_srid_ptr = nullptr;
  if (arg1_srid_out == 4326 &&
      static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kINTERSECTION) {
    // Use the best (location-based) planar transform to project 4326 argument before
    // running geos operation, back-project the result of the operation to 4326
    // TODO: Turn on automatic planar transform for Intersection, other binary ops
    // best_planar_srid_ptr = &best_planar_srid;
  }
  size_t wkb1_size{};
  WkbUniquePtr wkb1_ptr(toWkb(&wkb1_size,
                              arg1_type,
                              arg1_coords,
                              arg1_coords_size,
                              arg1_meta1,
                              arg1_meta1_size,
                              arg1_meta2,
                              arg1_meta2_size,
                              arg1_ic,
                              arg1_srid_in,
                              arg1_srid_out,
                              best_planar_srid_ptr));
  if (!wkb1_ptr) {
    return false;
  }
  size_t wkb2_size{};
  WkbUniquePtr wkb2_ptr(toWkb(&wkb2_size,
                              arg2_type,
                              arg2_coords,
                              arg2_coords_size,
                              arg2_meta1,
                              arg2_meta1_size,
                              arg2_meta2,
                              arg2_meta2_size,
                              arg2_ic,
                              arg2_srid_in,
                              arg2_srid_out,
                              best_planar_srid_ptr));
  if (!wkb2_ptr) {
    return false;
  }
  auto status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb1_ptr.get(), wkb1_size);
  if (g1) {
    auto* g2 = GEOSGeomFromWKB_buf_r(context, wkb2_ptr.get(), wkb2_size);
    if (g2) {
      GEOSGeometry* g = nullptr;
      if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kINTERSECTION) {
        g = GEOSIntersection_r(context, g1, g2);
      } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kDIFFERENCE) {
        g = GEOSDifference_r(context, g1, g2);
      } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kUNION) {
        g = GEOSUnion_r(context, g1, g2);
      }
      g = postprocess(context, g);
      if (g) {
        size_t wkb_size = 0ULL;
        WkbUniquePtr wkb_unique_ptr(GEOSGeomToWKB_buf_r(context, g, &wkb_size));
        if (wkb_unique_ptr.get() && wkb_size) {
          status = fromWkb(wkb_unique_ptr.get(),
                           wkb_size,
                           result_type,
                           result_coords,
                           result_coords_size,
                           result_meta1,
                           result_meta1_size,
                           result_meta2,
                           result_meta2_size,
                           /* result_srid_in = */ arg1_srid_out,
                           result_srid_out,
                           best_planar_srid_ptr);
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

extern "C" RUNTIME_EXPORT bool Geos_Wkb_Wkb_Predicate(
    int op,
    int arg1_type,
    int8_t* arg1_coords,
    int64_t arg1_coords_size,
    int32_t* arg1_meta1,
    int64_t arg1_meta1_size,
    int32_t* arg1_meta2,
    int64_t arg1_meta2_size,
    // TODO: add meta3 args to support generic geometries
    int32_t arg1_ic,
    int32_t arg1_srid_in,
    int32_t arg1_srid_out,
    int arg2_type,
    int8_t* arg2_coords,
    int64_t arg2_coords_size,
    int32_t* arg2_meta1,
    int64_t arg2_meta1_size,
    int32_t* arg2_meta2,
    int64_t arg2_meta2_size,
    // TODO: add meta3 args to support generic geometries
    int32_t arg2_ic,
    int32_t arg2_srid_in,
    int32_t arg2_srid_out,
    bool* result) {
#ifndef __CUDACC__
  size_t wkb1_size{};
  WkbUniquePtr wkb1_ptr(toWkb(&wkb1_size,
                              arg1_type,
                              arg1_coords,
                              arg1_coords_size,
                              arg1_meta1,
                              arg1_meta1_size,
                              arg1_meta2,
                              arg1_meta2_size,
                              arg1_ic,
                              arg1_srid_in,
                              arg1_srid_out,
                              nullptr));
  if (!wkb1_ptr) {
    return false;
  }
  size_t wkb2_size{};
  WkbUniquePtr wkb2_ptr(toWkb(&wkb2_size,
                              arg2_type,
                              arg2_coords,
                              arg2_coords_size,
                              arg2_meta1,
                              arg2_meta1_size,
                              arg2_meta2,
                              arg2_meta2_size,
                              arg2_ic,
                              arg2_srid_in,
                              arg2_srid_out,
                              nullptr));
  if (!wkb2_ptr) {
    return false;
  }
  auto status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb1_ptr.get(), wkb1_size);
  if (g1) {
    auto* g2 = GEOSGeomFromWKB_buf_r(context, wkb2_ptr.get(), wkb2_size);
    if (g2) {
      if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kEQUALS) {
        if (arg1_ic != arg2_ic &&
            (arg1_ic == COMPRESSION_GEOINT32 || arg2_ic == COMPRESSION_GEOINT32)) {
          *result = GEOSEqualsExact_r(context, g1, g2, TOLERANCE_GEOINT32);
        } else {
          *result = GEOSEquals_r(context, g1, g2);
        }
        status = true;
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

extern "C" RUNTIME_EXPORT bool Geos_Wkb_double(
    int op,
    int arg1_type,
    int8_t* arg1_coords,
    int64_t arg1_coords_size,
    int32_t* arg1_meta1,
    int64_t arg1_meta1_size,
    int32_t* arg1_meta2,
    int64_t arg1_meta2_size,
    // TODO: add meta3 args to support generic geometries
    int32_t arg1_ic,
    int32_t arg1_srid_in,
    int32_t arg1_srid_out,
    double arg2,
    // TODO: add transform args
    int* result_type,
    int8_t** result_coords,
    int64_t* result_coords_size,
    int32_t** result_meta1,
    int64_t* result_meta1_size,
    int32_t** result_meta2,
    int64_t* result_meta2_size,
    // TODO: add support for output compression
    int32_t result_srid_out) {
#ifndef __CUDACC__
  int32_t best_planar_srid;
  int32_t* best_planar_srid_ptr = nullptr;
  if (arg1_srid_out == 4326 &&
      static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kBUFFER && arg2 != 0.0) {
    // Use the best (location-based) planar transform to project 4326 argument before
    // running geos operation, back-project the result of the operation to 4326
    best_planar_srid_ptr = &best_planar_srid;
  }
  size_t wkb_size{};
  WkbUniquePtr wkb_ptr(toWkb(&wkb_size,
                             arg1_type,
                             arg1_coords,
                             arg1_coords_size,
                             arg1_meta1,
                             arg1_meta1_size,
                             arg1_meta2,
                             arg1_meta2_size,
                             arg1_ic,
                             arg1_srid_in,
                             arg1_srid_out,
                             best_planar_srid_ptr));
  if (!wkb_ptr) {
    return false;
  }
  auto status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb_ptr.get(), wkb_size);
  if (g1) {
    GEOSGeometry* g = nullptr;
    if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kBUFFER) {
      if (arg2 != 0.0) {
        int quadsegs = 8;  // default
        g = GEOSBuffer_r(context, g1, arg2, quadsegs);
      } else {
        g = GEOSGeom_clone_r(context, g1);
      }
    } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kCONCAVEHULL) {
#if (GEOS_VERSION_MAJOR > 3) || (GEOS_VERSION_MAJOR == 3 && GEOS_VERSION_MINOR >= 11)
      constexpr bool allow_holes = false;
      g = GEOSConcaveHull_r(context, g1, arg2, allow_holes);
#endif
    } else if (static_cast<GeoBase::GeoOp>(op) == GeoBase::GeoOp::kCONVEXHULL) {
      g = GEOSConvexHull_r(context, g1);
    }
    g = postprocess(context, g);
    if (g) {
      size_t wkb_size = 0ULL;
      WkbUniquePtr wkb_unique_ptr(GEOSGeomToWKB_buf_r(context, g, &wkb_size));
      if (wkb_unique_ptr.get() && wkb_size) {
        // Back-project the result from planar to 4326 if necessary
        status = fromWkb(wkb_unique_ptr.get(),
                         wkb_size,
                         result_type,
                         result_coords,
                         result_coords_size,
                         result_meta1,
                         result_meta1_size,
                         result_meta2,
                         result_meta2_size,
                         /* result_srid_in = */ arg1_srid_out,
                         result_srid_out,
                         best_planar_srid_ptr);
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

extern "C" RUNTIME_EXPORT bool Geos_Wkb(
    int op,
    int arg_type,
    int8_t* arg_coords,
    int64_t arg_coords_size,
    int32_t* arg_meta1,
    int64_t arg_meta1_size,
    int32_t* arg_meta2,
    int64_t arg_meta2_size,
    // TODO: add meta3 args to support generic geometries
    int32_t arg_ic,
    int32_t arg_srid_in,
    int32_t arg_srid_out,
    bool* result) {
#ifndef __CUDACC__
  if (!result) {
    return false;
  }
  size_t wkb_size{};
  WkbUniquePtr wkb_ptr(toWkb(&wkb_size,
                             arg_type,
                             arg_coords,
                             arg_coords_size,
                             arg_meta1,
                             arg_meta1_size,
                             arg_meta2,
                             arg_meta2_size,
                             arg_ic,
                             arg_srid_in,
                             arg_srid_out,
                             nullptr));
  if (!wkb_ptr) {
    return false;
  }
  bool status = false;
  auto context = create_context();
  if (!context) {
    return status;
  }
  auto* g1 = GEOSGeomFromWKB_buf_r(context, wkb_ptr.get(), wkb_size);
  if (g1) {
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
