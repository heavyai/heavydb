/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "Geospatial/GeosValidation.h"

#if defined(ENABLE_GEOS) && !defined(_WIN32)

#include <dlfcn.h>
#include <geos_c.h>
#include <mutex>

#include "Logger/Logger.h"

#ifndef GEOS_LIBRARY_FILENAME
#error Configuration should include GEOS library file name
#endif

// this is externed in DBHandler.cpp and NativeCodegen.cpp
std::unique_ptr<std::string> g_libgeos_so_filename(
    new std::string(GEOS_LIBRARY_FILENAME));

namespace Geospatial {

using PFN_GEOS_init_r = GEOSContextHandle_t (*)();
using PFN_GEOS_finish_r = void (*)(GEOSContextHandle_t);
using PFN_GEOSisValidDetail_r =
    char (*)(GEOSContextHandle_t, const GEOSGeometry*, int, char**, GEOSGeometry**);
using PFN_GEOSGeomFromWKB_buf_r = GEOSGeometry* (*)(GEOSContextHandle_t,
                                                    const unsigned char*,
                                                    size_t);
using PFN_GEOSGeom_destroy_r = void (*)(GEOSContextHandle_t, GEOSGeometry*);
using PFN_GEOSFree_r = void (*)(GEOSContextHandle_t, void*);

static bool geos_can_validate = true;
static void* geos_dso_handle = nullptr;

static std::mutex geos_dso_mutex;

static PFN_GEOS_init_r pfn_GEOS_init_r = nullptr;
static PFN_GEOS_finish_r pfn_GEOS_finish_r = nullptr;
static PFN_GEOSisValidDetail_r pfn_GEOSisValidDetail_r = nullptr;
static PFN_GEOSGeomFromWKB_buf_r pfn_GEOSGeomFromWKB_buf_r = nullptr;
static PFN_GEOSGeom_destroy_r pfn_GEOSGeom_destroy_r = nullptr;
static PFN_GEOSFree_r pfn_GEOSFree_r = nullptr;

bool geos_init() {
  // take lock
  std::lock_guard<std::mutex> guard(geos_dso_mutex);

  // already failed
  if (!geos_can_validate) {
    return false;
  }

  // already initialized
  if (geos_dso_handle) {
    return true;
  }

  // validate filename
  if (!g_libgeos_so_filename) {
    LOG(WARNING)
        << "GEOS dynamic library filename unspecified. Geometry validation unavailable.";
    geos_can_validate = false;
    return false;
  }

  // attempt to load DSO
  auto const* geos_dso_filename = g_libgeos_so_filename->c_str();
  geos_dso_handle = dlopen(geos_dso_filename, RTLD_NOW | RTLD_LOCAL);
  if (!geos_dso_handle) {
    LOG(ERROR) << "Failed to load GEOS library. To use Geometry Validation, ensure that "
                  "the GEOS library files are separately installed and that their "
                  "location is set in $LD_LIBRARY_PATH in the server environment.";
    geos_can_validate = false;
    return false;
  }

  // fetch all required function pointers
  pfn_GEOS_init_r = (PFN_GEOS_init_r)dlsym(geos_dso_handle, "GEOS_init_r");
  pfn_GEOS_finish_r = (PFN_GEOS_finish_r)dlsym(geos_dso_handle, "GEOS_finish_r");
  pfn_GEOSisValidDetail_r =
      (PFN_GEOSisValidDetail_r)dlsym(geos_dso_handle, "GEOSisValidDetail_r");
  pfn_GEOSGeomFromWKB_buf_r =
      (PFN_GEOSGeomFromWKB_buf_r)dlsym(geos_dso_handle, "GEOSGeomFromWKB_buf_r");
  pfn_GEOSGeom_destroy_r =
      (PFN_GEOSGeom_destroy_r)dlsym(geos_dso_handle, "GEOSGeom_destroy_r");
  pfn_GEOSFree_r = (PFN_GEOSFree_r)dlsym(geos_dso_handle, "GEOSFree_r");

  if (!pfn_GEOS_init_r || !pfn_GEOS_finish_r || !pfn_GEOSisValidDetail_r ||
      !pfn_GEOSGeomFromWKB_buf_r || !pfn_GEOSGeom_destroy_r || !pfn_GEOSFree_r) {
    LOG(WARNING) << "Failed to dynamically load required GEOS function pointers. Check "
                    "GEOS DSO version. Geometry validation unavailable.";
    geos_can_validate = false;
    return false;
  }

  // done
  return true;
}

bool geos_validation_available() {
  return geos_init();
}

bool geos_validate_wkb(const unsigned char* wkb, const size_t wkb_size) {
  // available?
  if (!geos_init()) {
    return true;
  }
  // validate
  auto* context = pfn_GEOS_init_r();
  CHECK(context);
  auto* geom = pfn_GEOSGeomFromWKB_buf_r(context, wkb, wkb_size);
  bool result = false;
  if (geom) {
    char* reason{};
    GEOSGeometry* location{};
    const int flags = 0;
    result = pfn_GEOSisValidDetail_r(context, geom, flags, &reason, &location) != 0;
    if (!result) {
      LOG(WARNING) << "GEOS invalid reason: " << reason;
    }
    pfn_GEOSFree_r(context, reason);
    pfn_GEOSGeom_destroy_r(context, location);
    pfn_GEOSGeom_destroy_r(context, geom);
  }
  pfn_GEOS_finish_r(context);
  return result;
}

}  // namespace Geospatial

#else

namespace Geospatial {

bool geos_validation_available() {
  return false;
}

bool geos_validate_wkb(const unsigned char* /* wkb */, const size_t /* wkb_size */) {
  return true;
}

}  // namespace Geospatial

#endif
