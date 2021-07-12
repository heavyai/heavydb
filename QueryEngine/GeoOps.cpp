/*
 * Copyright 2021 OmniSci, Inc.
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
 * @file    GeoOps.cpp
 * @author  Alex Baden <alexb@omnisci.com>
 * @brief   Functions to support geospatial operations used by the executor.
 *
 **/

#include "../Geospatial/Compression.h"
#include "../Geospatial/Transforms.h"
#include "../Shared/funcannotations.h"
#include "TypePunning.h"

#ifdef EXECUTE_INCLUDE

extern "C" DEVICE RUNTIME_EXPORT double decompress_x_coord_geoint(const int32_t coord) {
  return Geospatial::decompress_longitude_coord_geoint32(coord);
}

extern "C" DEVICE RUNTIME_EXPORT double decompress_y_coord_geoint(const int32_t coord) {
  return Geospatial::decompress_lattitude_coord_geoint32(coord);
}

extern "C" DEVICE RUNTIME_EXPORT int32_t compress_x_coord_geoint(const double coord) {
  return static_cast<int32_t>(Geospatial::compress_longitude_coord_geoint32(coord));
}

extern "C" DEVICE RUNTIME_EXPORT int32_t compress_y_coord_geoint(const double coord) {
  return static_cast<int32_t>(Geospatial::compress_lattitude_coord_geoint32(coord));
}

#endif
