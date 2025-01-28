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

#include <iomanip>
#include <iostream>
#include <limits>

#include <h3/h3api.h>

namespace Geospatial {

#define MAX_INDEX_STR_LEN 20

#define MIN_RESOLUTION 0
#define MAX_RESOLUTION 15

int64_t H3_LonLatToCell(const double lon, const double lat, const int32_t resolution) {
  if (resolution < MIN_RESOLUTION || resolution > MAX_RESOLUTION) {
    return H3_NULL;
  }
  LatLng ll{degsToRads(lat), degsToRads(lon)};
  H3Index cell{};
  latLngToCell(&ll, resolution, &cell);
  return shared::reinterpret_bits<int64_t>(cell);
}

std::pair<double, double> H3_CellToLonLat(const int64_t cell) {
  LatLng ll;
  cellToLatLng(shared::reinterpret_bits<H3Index>(cell), &ll);
  return {radsToDegs(ll.lng), radsToDegs(ll.lat)};
}

std::string H3_CellToString(const int64_t cell) {
  char str[MAX_INDEX_STR_LEN];
  if (h3ToString(shared::reinterpret_bits<H3Index>(cell), str, MAX_INDEX_STR_LEN) !=
      E_SUCCESS) {
    return "NULL";
  }
  return std::string(str);
}

int64_t H3_StringToCell(const std::string& str) {
  H3Index cell{};
  stringToH3(str.c_str(), &cell);
  return shared::reinterpret_bits<int64_t>(cell);
}

int64_t H3_CellToParent(const int64_t cell, const int32_t resolution) {
  H3Index parent_cell{};
  cellToParent(shared::reinterpret_bits<H3Index>(cell), resolution, &parent_cell);
  return shared::reinterpret_bits<int64_t>(parent_cell);
}

bool H3_IsValidCell(const int64_t cell) {
  return (isValidCell(shared::reinterpret_bits<H3Index>(cell)) != 0);
}

std::string H3_CellToBoundary_WKT(const int64_t cell) {
  CellBoundary cb;
  if (cellToBoundary(shared::reinterpret_bits<H3Index>(cell), &cb) != E_SUCCESS) {
    return "POLYGON EMPTY";
  }
  std::ostringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::fixed;
  ss << "POLYGON((";
  for (int i = 0; i < cb.numVerts; i++) {
    auto const& point = cb.verts[i];
    if (i > 0) {
      ss << ", ";
    }
    ss << radsToDegs(point.lng) << " " << radsToDegs(point.lat);
  }
  ss << "))";
  return ss.str();
}

}  // namespace Geospatial
