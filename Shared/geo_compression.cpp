/*
 * Copyright 2020 OmniSci, Inc.
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

#include "Shared/geo_compression.h"

namespace geospatial {

uint64_t compress_coord(double coord, const SQLTypeInfo& ti, bool x) {
  if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
    return x ? Geo_namespace::compress_longitude_coord_geoint32(coord)
             : Geo_namespace::compress_lattitude_coord_geoint32(coord);
  }
  return *reinterpret_cast<uint64_t*>(may_alias_ptr(&coord));
}

uint64_t compress_null_point(const SQLTypeInfo& ti, bool x) {
  if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
    return x ? Geo_namespace::compress_null_point_longitude_geoint32()
             : Geo_namespace::compress_null_point_lattitude_geoint32();
  }
  double n = x ? NULL_ARRAY_DOUBLE : NULL_DOUBLE;
  auto u = *reinterpret_cast<uint64_t*>(may_alias_ptr(&n));
  return u;
}

// Compress non-NULL geo coords; and also NULL POINT coords (special case)
std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti) {
  CHECK(!coords.empty()) << "Coord compression received no data";
  bool is_null_point = false;
  if (!ti.get_notnull()) {
    is_null_point = (ti.get_type() == kPOINT && coords[0] == NULL_ARRAY_DOUBLE);
  }
  std::vector<uint8_t> compressed_coords;
  bool x = true;
  bool is_geoint32 =
      (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32);
  size_t coord_data_size = (is_geoint32) ? (ti.get_comp_param() / 8) : sizeof(double);
  for (auto coord : coords) {
    uint64_t coord_data;
    if (is_null_point) {
      coord_data = compress_null_point(ti, x);
    } else {
      if (ti.get_output_srid() == 4326) {
        if (x) {
          if (coord < -180.0 || coord > 180.0) {
            throw std::runtime_error("WGS84 longitude " + std::to_string(coord) +
                                     " is out of bounds");
          }
        } else {
          if (coord < -90.0 || coord > 90.0) {
            throw std::runtime_error("WGS84 latitude " + std::to_string(coord) +
                                     " is out of bounds");
          }
        }
      }
      if (is_geoint32) {
        coord_data = compress_coord(coord, ti, x);
      } else {
        auto coord_data_ptr = reinterpret_cast<uint64_t*>(&coord);
        coord_data = *coord_data_ptr;
      }
    }
    for (size_t i = 0; i < coord_data_size; i++) {
      compressed_coords.push_back(coord_data & 0xFF);
      coord_data >>= 8;
    }
    x = !x;
  }
  return compressed_coords;
}

}  // namespace geospatial
