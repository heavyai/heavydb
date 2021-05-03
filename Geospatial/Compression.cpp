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

#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "QueryEngine/TypePunning.h"

namespace Geospatial {

int32_t get_compression_scheme(const SQLTypeInfo& ti) {
  if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
    return COMPRESSION_GEOINT32;
  }
  if (ti.get_compression() != kENCODING_NONE) {
    throw std::runtime_error("Invalid compression");
  }
  return COMPRESSION_NONE;
}

uint64_t compress_coord(double coord, const SQLTypeInfo& ti, bool x) {
  if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
    return x ? Geospatial::compress_longitude_coord_geoint32(coord)
             : Geospatial::compress_lattitude_coord_geoint32(coord);
  }
  return *reinterpret_cast<uint64_t*>(may_alias_ptr(&coord));
}

uint64_t compress_null_point(const SQLTypeInfo& ti, bool x) {
  if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
    return x ? Geospatial::compress_null_point_longitude_geoint32()
             : Geospatial::compress_null_point_lattitude_geoint32();
  }
  double n = x ? NULL_ARRAY_DOUBLE : NULL_DOUBLE;
  auto u = *reinterpret_cast<uint64_t*>(may_alias_ptr(&n));
  return u;
}

// Compress non-NULL geo coords; and also NULL POINT coords (special case)
std::vector<uint8_t> compress_coords(const std::vector<double>& coords,
                                     const SQLTypeInfo& ti) {
  CHECK(!coords.empty()) << "Coord compression received no data";
  bool is_null_point = false;
  if (!ti.get_notnull()) {
    is_null_point = (ti.get_type() == kPOINT && coords[0] == NULL_ARRAY_DOUBLE);
  }

  bool x = true;
  bool is_geoint32 =
      (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32);
  size_t coord_data_size = (is_geoint32) ? (ti.get_comp_param() / 8) : sizeof(double);
  std::vector<uint8_t> compressed_coords;
  compressed_coords.reserve(coords.size() * coord_data_size);
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

template <typename T>
void unpack_geo_vector(std::vector<T>& output, const int8_t* input_ptr, const size_t sz) {
  if (sz == 0) {
    return;
  }
  auto elems = reinterpret_cast<const T*>(input_ptr);
  CHECK_EQ(size_t(0), sz % sizeof(T));
  const size_t num_elems = sz / sizeof(T);
  output.resize(num_elems);
  for (size_t i = 0; i < num_elems; i++) {
    output[i] = elems[i];
  }
}

template <>
void unpack_geo_vector<int32_t>(std::vector<int32_t>& output,
                                const int8_t* input_ptr,
                                const size_t sz) {
  if (sz == 0) {
    return;
  }
  auto elems = reinterpret_cast<const int32_t*>(input_ptr);
  CHECK_EQ(size_t(0), sz % sizeof(int32_t));
  const size_t num_elems = sz / sizeof(int32_t);
  output.resize(num_elems);
  for (size_t i = 0; i < num_elems; i++) {
    output[i] = elems[i];
  }
}

template <typename T>
void decompress_geo_coords_geoint32(std::vector<T>& dec,
                                    const int8_t* enc,
                                    const size_t sz) {
  if (sz == 0) {
    return;
  }
  const auto compressed_coords = reinterpret_cast<const int32_t*>(enc);
  const auto num_coords = sz / sizeof(int32_t);
  dec.resize(num_coords);
  for (size_t i = 0; i < num_coords; i += 2) {
    dec[i] = Geospatial::decompress_longitude_coord_geoint32(compressed_coords[i]);
    dec[i + 1] =
        Geospatial::decompress_lattitude_coord_geoint32(compressed_coords[i + 1]);
  }
}

template <>
std::shared_ptr<std::vector<double>> decompress_coords<double, SQLTypeInfo>(
    const SQLTypeInfo& geo_ti,
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

template <>
std::shared_ptr<std::vector<double>> decompress_coords<double, int32_t>(
    const int32_t& ic,
    const int8_t* coords,
    const size_t coords_sz) {
  auto decompressed_coords_ptr = std::make_shared<std::vector<double>>();
  if (ic == COMPRESSION_GEOINT32) {
    decompress_geo_coords_geoint32(*decompressed_coords_ptr, coords, coords_sz);
  } else {
    CHECK_EQ(ic, COMPRESSION_NONE);
    unpack_geo_vector(*decompressed_coords_ptr, coords, coords_sz);
  }
  return decompressed_coords_ptr;
}

bool is_null_point(const SQLTypeInfo& geo_ti,
                   const int8_t* coords,
                   const size_t coords_sz) {
  if (geo_ti.get_type() == kPOINT && !geo_ti.get_notnull()) {
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      if (geo_ti.get_comp_param() == 32) {
        return Geospatial::is_null_point_longitude_geoint32(*((int32_t*)coords));
      }
    } else {
      CHECK_EQ(geo_ti.get_compression(), kENCODING_NONE);
      return *((double*)coords) == NULL_ARRAY_DOUBLE;
    }
  }
  return false;
}

}  // namespace Geospatial
