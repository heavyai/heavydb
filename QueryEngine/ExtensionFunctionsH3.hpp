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

#ifndef __CUDACC__

//
// H3_PointToCell(POINT p) -> BIGINT
//

EXTENSION_NOINLINE int64_t H3_PointToCell__cpu_(RowFunctionManager& mgr,
                                                GeoPoint& p,
                                                int32_t resolution) {
  const int32_t psize = p.getSize();
  const int32_t ic = p.getCompression();
  const int32_t isr = p.getInputSrid();
  const int32_t osr = p.getOutputSrid();
  const double lon = ST_X_Point(p.ptr, psize, ic, isr, osr);
  const double lat = ST_Y_Point(p.ptr, psize, ic, isr, osr);
  return Geospatial::H3_LonLatToCell(lon, lat, resolution);
}

//
// H3_LonLatToCell(DOUBLE lon, DOUBLE lat) -> BIGINT
//

EXTENSION_NOINLINE int64_t H3_LonLatToCell__cpu_(RowFunctionManager& mgr,
                                                 double lon,
                                                 double lat,
                                                 int32_t resolution) {
  return Geospatial::H3_LonLatToCell(lon, lat, resolution);
}

//
// H3_CellToLon/Lat(BIGINT cell) -> DOUBLE
//

EXTENSION_NOINLINE double H3_CellToLon__cpu_(RowFunctionManager& mgr, int64_t cell) {
  return Geospatial::H3_CellToLonLat(cell).first;
}

EXTENSION_NOINLINE double H3_CellToLat__cpu_(RowFunctionManager& mgr, int64_t cell) {
  return Geospatial::H3_CellToLonLat(cell).second;
}

//
// H3_CellToString(BIGINT cell) -> TEXT
//

EXTENSION_NOINLINE TextEncodingNone
H3_CellToString_TEXT_NONE__cpu_(RowFunctionManager& mgr, int64_t cell) {
  auto const cell_string = Geospatial::H3_CellToString(cell);
  return TextEncodingNone(mgr, cell_string);
}

EXTENSION_NOINLINE TextEncodingDict H3_CellToString_TEXT__cpu_(RowFunctionManager& mgr,
                                                               int64_t cell) {
  auto const cell_string = Geospatial::H3_CellToString(cell);
  return mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, cell_string);
}

//
// H3_StringToCell(TEXT hex) -> BIGINT
//

EXTENSION_NOINLINE int64_t H3_StringToCell__1__cpu_(RowFunctionManager& mgr,
                                                    TextEncodingNone none_str) {
  return Geospatial::H3_StringToCell(none_str.getString());
}

EXTENSION_NOINLINE int64_t H3_StringToCell__2__cpu_(RowFunctionManager& mgr,
                                                    TextEncodingDict dict_str) {
  std::string str = mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), dict_str);
  return Geospatial::H3_StringToCell(str);
}

//
// H3_CellToParent(BIGINT cell) -> BIGINT
//

EXTENSION_NOINLINE int64_t H3_CellToParent__cpu_(RowFunctionManager& mgr,
                                                 int64_t cell,
                                                 int32_t resolution) {
  return Geospatial::H3_CellToParent(cell, resolution);
}

//
// H3_IsValidCell(BIGINT cell) -> BOOL
//

EXTENSION_NOINLINE bool H3_IsValidCell__cpu_(RowFunctionManager& mgr, int64_t cell) {
  return Geospatial::H3_IsValidCell(cell);
}

//
// H3_CellToBoundary_WKT(BIGINT cell) -> TEXT
//

EXTENSION_NOINLINE TextEncodingDict H3_CellToBoundary_WKT__cpu_(RowFunctionManager& mgr,
                                                                int64_t cell) {
  auto const wkt_string = Geospatial::H3_CellToBoundary_WKT(cell);
  return mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, wkt_string);
}

//
// H3_CellToBoundary(BIGINT cell) -> POLYGON
// H3_CellToPoint(BIGINT cell) -> POINT
//
// These are implemented in H3Runtime.cpp
//

#endif  // __CUDACC__
