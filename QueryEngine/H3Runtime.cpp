/*
 * Copyright 2025 HEAVY.AI, Inc.
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

#ifndef __CUDACC__

#include "QueryEngine/H3Runtime.h"

#include "Geospatial/H3Shim.h"
#include "Geospatial/Types.h"

#endif

extern "C" RUNTIME_EXPORT bool H3_CellToBoundary_POLYGON(
    const int64_t cell,
    int* result_type,
    int8_t** result_coords,
    int64_t* result_coords_size,
    int32_t** result_ring_sizes,
    int64_t* result_ring_sizes_size) {
#ifndef __CUDACC__
  // fetch coords and determine ring size
  auto const coords = Geospatial::H3_CellToBoundary_POLYGON(cell);
  auto const num_points = coords.size() / 2;
  if (num_points == 0) {
    return false;
  }
  std::vector<int32_t> ring_sizes{static_cast<int32_t>(num_points)};

  // POLYGON
  *result_type = static_cast<int>(kPOLYGON);

  // coords
  *result_coords = nullptr;
  int64_t coords_size = coords.size() * sizeof(double);
  auto* coords_buf = malloc(coords_size);
  if (!coords_buf) {
    return false;
  }
  std::memcpy(coords_buf, coords.data(), coords_size);
  *result_coords = reinterpret_cast<int8_t*>(coords_buf);
  *result_coords_size = coords_size;

  // ring sizes
  *result_ring_sizes = nullptr;
  int64_t ring_sizes_size = ring_sizes.size() * sizeof(int32_t);
  auto* ring_sizes_buf = malloc(ring_sizes_size);
  if (!ring_sizes_buf) {
    free(coords_buf);
    return false;
  }
  std::memcpy(ring_sizes_buf, ring_sizes.data(), ring_sizes_size);
  *result_ring_sizes = reinterpret_cast<int32_t*>(ring_sizes_buf);
  *result_ring_sizes_size = ring_sizes.size();

  return true;
#else
  return false;
#endif
}

extern "C" RUNTIME_EXPORT bool H3_CellToPoint(const int64_t cell,
                                              int* result_type,
                                              int8_t** result_coords,
                                              int64_t* result_coords_size) {
#ifndef __CUDACC__
  // coords
  auto const lon_lat = Geospatial::H3_CellToLonLat(cell);
  const std::array<double, 2> coords = {lon_lat.first, lon_lat.second};

  // POINT
  *result_type = static_cast<int>(kPOINT);

  // coords
  *result_coords = nullptr;
  int64_t coords_size = coords.size() * sizeof(double);
  auto* coords_buf = malloc(coords_size);
  if (!coords_buf) {
    return false;
  }
  std::memcpy(coords_buf, coords.data(), coords_size);
  *result_coords = reinterpret_cast<int8_t*>(coords_buf);
  *result_coords_size = coords_size;

  return true;
#else
  return false;
#endif
}
