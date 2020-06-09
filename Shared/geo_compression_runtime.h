/*
 * Copyright 2018 OmniSci, Inc.
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
 * @file    geo_compression_runtime.h
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Compression / decompression routines for geospatial coordinates. Suitable for
 * inclusion by the LLVM codegen runtime.
 *
 */

#ifndef SHARED_GEO_COMPRESSION_H
#define SHARED_GEO_COMPRESSION_H

#include "funcannotations.h"

#define COMPRESSION_NONE 0
#define COMPRESSION_GEOINT32 1
#define COMPRESSION_GEOBBINT32 2
#define COMPRESSION_GEOBBINT16 3
#define COMPRESSION_GEOBBINT8 4

#define TOLERANCE_DEFAULT 0.000000001
#define TOLERANCE_GEOINT32 0.0000001

namespace Geo_namespace {

DEVICE inline double decompress_longitude_coord_geoint32(const int32_t compressed) {
  // decompress longitude: -2,147,483,647..2,147,483,647  --->  -180..180
  return static_cast<double>(compressed) *
         8.3819031754424345e-08;  // (180.0 / 2147483647.0)
}

DEVICE inline double decompress_lattitude_coord_geoint32(const int32_t compressed) {
  // decompress latitude: -2,147,483,647..2,147,483,647  --->  -90..90
  return static_cast<double>(compressed) *
         4.1909515877212172e-08;  // // (90.0 / 2147483647.0)
}

DEVICE inline bool is_null_point_longitude_geoint32(const int32_t compressed) {
  // check compressed null point longitude: -2,147,483,648  --->  NULL
  return (*reinterpret_cast<const uint32_t*>(&compressed) == 0x80000000U);
}

DEVICE inline bool is_null_point_lattitude_geoint32(const int32_t compressed) {
  // check compressed null point latitude: -2,147,483,648  --->  NULL
  return (*reinterpret_cast<const uint32_t*>(&compressed) == 0x80000000U);
}

DEVICE inline uint64_t compress_longitude_coord_geoint32(const double coord) {
  // compress longitude: -180..180  --->  -2,147,483,647..2,147,483,647
  int32_t compressed_coord = static_cast<int32_t>(coord * (2147483647.0 / 180.0));
  return static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&compressed_coord));
}

DEVICE inline uint64_t compress_lattitude_coord_geoint32(const double coord) {
  // compress latitude: -90..90  --->  -2,147,483,647..2,147,483,647
  int32_t compressed_coord = static_cast<int32_t>(coord * (2147483647.0 / 90.0));
  return static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&compressed_coord));
}

DEVICE constexpr uint64_t compress_null_point_longitude_geoint32() {
  // compress null point longitude: NULL  --->  -2,147,483,648
  return 0x0000000080000000ULL;
}

DEVICE constexpr uint64_t compress_null_point_lattitude_geoint32() {
  // compress null point latitude: NULL  --->  -2,147,483,648
  return 0x0000000080000000ULL;
}

};  // namespace Geo_namespace

#endif  // SHARED_GEO_COMPRESSION_H
