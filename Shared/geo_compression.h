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
 * @file    geo_compression.h
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Compression / decompression routines for geospatial coordinates.
 *
 */

#ifndef SHARED_GEO_COMPRESSION_H
#define SHARED_GEO_COMPRESSION_H

#include "funcannotations.h"

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

};  // namespace Geo_namespace

#endif  // SHARED_GEO_COMPRESSION_H
