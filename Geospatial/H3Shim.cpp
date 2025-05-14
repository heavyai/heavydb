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

#include "Geospatial/H3Shim.h"
#include "Shared/misc.h"

#include <cfloat>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <h3/h3api.h>

namespace Geospatial {

#define MAX_INDEX_STR_LEN 20

#define MIN_RESOLUTION 0
#define MAX_RESOLUTION 15

#define NULL_DOUBLE DBL_MIN

int64_t H3_LonLatToCell(const double lon, const double lat, const int32_t resolution) {
  if (resolution < MIN_RESOLUTION || resolution > MAX_RESOLUTION) {
    throw std::runtime_error("H3_LonLatToCell: invalid resolution (" +
                             std::to_string(resolution) + ")");
  }
  if (lon == NULL_DOUBLE || lat == NULL_DOUBLE) {
    return H3_NULL;
  }
  LatLng ll{degsToRads(lat), degsToRads(lon)};
  H3Index cell{};
  latLngToCell(&ll, resolution, &cell);
  return shared::reinterpret_bits<int64_t>(cell);
}

std::pair<double, double> H3_CellToLonLat(const int64_t cell_in) {
  auto const cell = shared::reinterpret_bits<H3Index>(cell_in);
  if (cell == H3_NULL) {
    return {NULL_DOUBLE, NULL_DOUBLE};
  }
  LatLng ll;
  cellToLatLng(cell, &ll);
  return {radsToDegs(ll.lng), radsToDegs(ll.lat)};
}

std::string H3_CellToString(const int64_t cell_in) {
  auto const cell = shared::reinterpret_bits<H3Index>(cell_in);
  if (cell == H3_NULL) {
    return "NULL";
  }
  char str[MAX_INDEX_STR_LEN];
  if (h3ToString(cell, str, MAX_INDEX_STR_LEN) != E_SUCCESS) {
    return "NULL";
  }
  return std::string(str);
}

int64_t H3_StringToCell(const std::string& str) {
  H3Index cell{};
  stringToH3(str.c_str(), &cell);
  return shared::reinterpret_bits<int64_t>(cell);
}

int64_t H3_CellToParent(const int64_t cell_in, const int32_t resolution) {
  if (resolution < MIN_RESOLUTION || resolution > MAX_RESOLUTION) {
    throw std::runtime_error("H3_CellToParent: invalid resolution (" +
                             std::to_string(resolution) + ")");
  }
  auto const cell = shared::reinterpret_bits<H3Index>(cell_in);
  if (cell == H3_NULL) {
    return shared::reinterpret_bits<int64_t>(H3_NULL);
  }
  H3Index parent_cell{};
  cellToParent(cell, resolution, &parent_cell);
  return shared::reinterpret_bits<int64_t>(parent_cell);
}

bool H3_IsValidCell(const int64_t cell_in) {
  auto const cell = shared::reinterpret_bits<H3Index>(cell_in);
  if (cell == H3_NULL) {
    return false;
  }
  return (isValidCell(cell) != 0);
}

std::vector<double> H3_CellToBoundary_POLYGON(const int64_t cell_in) {
  auto const cell = shared::reinterpret_bits<H3Index>(cell_in);
  if (cell == H3_NULL) {
    return {};
  }
  CellBoundary cb;
  if (cellToBoundary(cell, &cb) != E_SUCCESS) {
    return {};
  }
  std::vector<double> coords(cb.numVerts * 2);
  int coord_index{};
  for (int i = 0; i < cb.numVerts; i++) {
    coords[coord_index++] = radsToDegs(cb.verts[i].lng);
    coords[coord_index++] = radsToDegs(cb.verts[i].lat);
  }
  return coords;
}

std::string H3_CellToBoundary_WKT(const int64_t cell_in) {
  auto const cell = shared::reinterpret_bits<H3Index>(cell_in);
  if (cell == H3_NULL) {
    return "POLYGON EMPTY";
  }
  CellBoundary cb;
  if (cellToBoundary(cell, &cb) != E_SUCCESS) {
    return "POLYGON EMPTY";
  }
  std::ostringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::fixed;
  ss << "POLYGON((";
  // repeat first point at end, per OGC and our standard behavior
  for (int i = 0; i <= cb.numVerts; i++) {
    auto const& point = cb.verts[i % cb.numVerts];
    if (i > 0) {
      ss << ", ";
    }
    ss << radsToDegs(point.lng) << " " << radsToDegs(point.lat);
  }
  ss << "))";
  return ss.str();
}

}  // namespace Geospatial
