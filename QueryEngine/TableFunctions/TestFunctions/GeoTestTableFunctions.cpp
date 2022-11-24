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

#include "TableFunctionsTesting.h"

/*
  This file contains testing geometry-related compile-time UDTFs.

  NOTE: This file currently has no GPU UDTFs. If any GPU UDTFs are
  added, it should be added to CUDA_TABLE_FUNCTION_FILES in CMakeLists.txt
 */

#ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_coords__cpu_(TableFunctionManager& mgr,
                                           const Column<GeoPoint>& points,
                                           Column<double>& xcoords,
                                           Column<double>& ycoords) {
  auto size = points.size();
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (points.isNull(i)) {
      xcoords.setNull(i);
      ycoords.setNull(i);
    } else {
      const auto point = points[i];
      xcoords[i] = point.x;
      ycoords[i] = point.y;
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t ct_shift__cpu_(TableFunctionManager& mgr,
                                          const Column<GeoPoint>& points,
                                          const double x,
                                          const double y,
                                          Column<GeoPoint>& shifted_points) {
  auto size = points.size();
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (points.isNull(i)) {
      shifted_points.setNull(i);
    } else {
      auto point = points[i];
      point.x += x;
      point.y += y;
      shifted_points.setItem(i, point);
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t ct_pointn__cpu_(TableFunctionManager& mgr,
                                           const Column<GeoLineString>& linestrings,
                                           int64_t n,
                                           Column<double>& xcoords,
                                           Column<double>& ycoords) {
  auto size = linestrings.size();
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (linestrings.isNull(i)) {
      xcoords.setNull(i);
      ycoords.setNull(i);
    } else {
      const auto point = linestrings[i][n - 1];  // n is one-based
      xcoords[i] = point.x;
      ycoords[i] = point.y;
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t ct_copy__cpu_(TableFunctionManager& mgr,
                                         const Column<GeoLineString>& linestrings,
                                         Column<GeoLineString>& copied_linestrings) {
  auto size = linestrings.size();
  mgr.set_output_item_values_total_number(0, linestrings.getNofValues());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (linestrings.isNull(i)) {
      copied_linestrings.setNull(i);
    } else {
      copied_linestrings[i] = linestrings[i];
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t ct_linestringn__cpu_(TableFunctionManager& mgr,
                                                const Column<GeoPolygon>& polygons,
                                                int64_t n,
                                                Column<GeoLineString>& linestrings) {
  auto size = polygons.size();
  mgr.set_output_item_values_total_number(0, polygons.getNofValues());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (polygons.isNull(i)) {
      linestrings.setNull(i);
    } else {
      linestrings.setItem(i, polygons[i], n - 1);
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t ct_make_polygon3__cpu_(TableFunctionManager& mgr,
                                                  const Column<GeoLineString>& rings,
                                                  const Column<GeoLineString>& holes1,
                                                  const Column<GeoLineString>& holes2,
                                                  Column<GeoPolygon>& polygons,
                                                  Column<int>& sizes) {
  auto size = rings.size();
  mgr.set_output_item_values_total_number(
      0, rings.getNofValues() + holes1.getNofValues() + holes2.getNofValues());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (rings.isNull(i)) {
      polygons.setNull(i);
      sizes.setNull(i);
    } else {
      std::vector<std::vector<double>> polygon_coords;

      polygon_coords.push_back(rings[i].toCoords());
      polygon_coords.push_back(holes1[i].toCoords());
      polygon_coords.push_back(holes2[i].toCoords());

      auto polygon = polygons[i];
      auto status = polygon.fromCoords(polygon_coords);

      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
      }
      int nofpoints = 0;
      for (int j = 0; j < polygon.size(); j++) {
        nofpoints += polygon.size(j);
      }
      sizes[i] = nofpoints;
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t ct_make_linestring2__cpu_(TableFunctionManager& mgr,
                                                     const Column<double>& x,
                                                     const Column<double>& y,
                                                     double dx,
                                                     double dy,
                                                     Column<GeoLineString>& linestrings) {
  auto size = x.size();
  mgr.set_output_item_values_total_number(0, size * 4);
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (x.isNull(i) || y.isNull(i)) {
      linestrings.setNull(i);
    } else {
      double line[4] = {x[i], y[i], x[i] + dx, y[i] + dy};
      linestrings.setItem(
          i, reinterpret_cast<const int8_t*>(&line[0]), 4 * sizeof(double));
    }
  }
  return size;
}

#endif  // #ifndef __CUDACC__