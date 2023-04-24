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

template <typename T>
NEVER_INLINE HOST int32_t ct_pointn__cpu_template(TableFunctionManager& mgr,
                                                  const Column<T>& points,
                                                  int64_t n,
                                                  Column<double>& xcoords,
                                                  Column<double>& ycoords) {
  auto size = points.size();
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (points.isNull(i)) {
      xcoords.setNull(i);
      ycoords.setNull(i);
    } else {
      const auto point = points[i][n - 1];  // n is one-based
      xcoords[i] = point.x;
      ycoords[i] = point.y;
    }
  }
  return size;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_pointn__cpu_template(TableFunctionManager& mgr,
                        const Column<GeoLineString>& points,
                        int64_t n,
                        Column<double>& xcoords,
                        Column<double>& ycoords);

template NEVER_INLINE HOST int32_t
ct_pointn__cpu_template(TableFunctionManager& mgr,
                        const Column<GeoMultiPoint>& points,
                        int64_t n,
                        Column<double>& xcoords,
                        Column<double>& ycoords);

// explicit instantiations

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
      int64_t sz = polygons[i].size();
      if (n < 1 || n > sz) {
        linestrings.setNull(i);
      } else {
        const auto poly = polygons[i];
        const auto ring = poly[n - 1];
        linestrings.setItem(i, ring);
      }
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
  // Initialize polygons
  int count_nulls = 0;
  for (int64_t i = 0; i < size; i++) {
    if (rings.isNull(i)) {
      polygons.setNull(i);
      sizes.setNull(i);
      count_nulls++;
    } else {
      std::vector<std::vector<double>> polygon_coords;

      polygon_coords.push_back(rings[i].toCoords());
      if (!holes1.isNull(i)) {
        polygon_coords.push_back(holes1[i].toCoords());
        if (!holes2.isNull(i)) {
          polygon_coords.push_back(holes2[i].toCoords());
        }
      }

      auto polygon = polygons[i];
      auto status = polygon.fromCoords(polygon_coords);

      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
      }
      int nofpoints = 0;
      for (size_t j = 0; j < polygon.size(); j++) {
        nofpoints += polygon.size(j);
      }
      sizes[i] = nofpoints;
    }
  }

  // Check polygons content
  if (count_nulls == 0) {
    return mgr.ERROR_MESSAGE("counting null test failed: count_nulls=" +
                             ::toString(count_nulls) + ", expected non-zero.");
  }

  for (int64_t i = 0; i < size; i++) {
    if (polygons.isNull(i)) {
      count_nulls--;
    } else {
      std::vector<std::vector<double>> polygon_coords;
      polygon_coords.push_back(rings[i].toCoords());
      if (!holes1.isNull(i)) {
        polygon_coords.push_back(holes1[i].toCoords());
      }
      if (!holes2.isNull(i)) {
        polygon_coords.push_back(holes2[i].toCoords());
      }

      // polygons[i] is Geo::Polygon instances
      // polygons[i][j] is Geo::LineString instances
      // polygons[i][j][k] is Geo::Point2D instances

      auto nof_lines = polygons[i].size();

      if (nof_lines != polygon_coords.size()) {
        return mgr.ERROR_MESSAGE(
            "polygon size test failed: nof_lines=" + ::toString(nof_lines) +
            ", expected " + ::toString(polygon_coords.size()) + ".");
      }
      std::vector<std::vector<double>> poly_coords = polygons[i].toCoords();
      if (nof_lines != poly_coords.size()) {
        return mgr.ERROR_MESSAGE(
            "polygon toCoords size test failed: poly_coords.size()=" +
            ::toString(poly_coords.size()) + ", expected " + ::toString(nof_lines) + ".");
      }

      auto poly = polygons[i];

      for (size_t j = 0; j < poly.size(); j++) {
        Geo::LineString line = poly[j];
        std::vector<double> line_coords = line.toCoords();
        auto nof_points = polygon_coords[j].size() / 2;
        if (poly.size(j) != nof_points) {
          return mgr.ERROR_MESSAGE("polygon linestring size test failed: poly.size(" +
                                   ::toString(j) + ")=" + ::toString(poly.size(j)) +
                                   ", expected " + ::toString(nof_points) + ".");
        }
        if (line.size() != nof_points) {
          return mgr.ERROR_MESSAGE("polygon linestring size test failed: line.size()=" +
                                   ::toString(line.size()) + ", expected " +
                                   ::toString(nof_points) + ".");
        }
        if (poly_coords[j].size() != nof_points * 2) {
          return mgr.ERROR_MESSAGE(
              "polygon linestring coords size test failed: poly_coords[j].size()=" +
              ::toString(poly_coords[j].size()) + ", expected " +
              ::toString(nof_points * 2) + ".");
        }
        if (line_coords.size() != nof_points * 2) {
          return mgr.ERROR_MESSAGE(
              "polygon linestring coords size test failed: line_coords.size()=" +
              ::toString(line_coords.size()) + ", expected " +
              ::toString(nof_points * 2) + ".");
        }
        for (size_t k = 0; k < nof_points; k++) {
          if (std::abs(polygon_coords[j][2 * k] - line_coords[2 * k]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "polygon linestring X coord test failed: line_coords[2*k]=" +
                ::toString(line_coords[2 * k]) + ", expected " +
                ::toString(polygon_coords[j][2 * k]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k] - poly_coords[j][2 * k]) > 1e-7) {
            return mgr.ERROR_MESSAGE("polygon X coord test failed: poly_coords[j][2*k]=" +
                                     ::toString(poly_coords[j][2 * k]) + ", expected " +
                                     ::toString(polygon_coords[j][2 * k]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k] - line[k].x) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "polygon linestring X coord test failed: line[k].x=" +
                ::toString(line[k].x) + ", expected " +
                ::toString(polygon_coords[j][2 * k]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k + 1] - line_coords[2 * k + 1]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "polygon linestring Y coord test failed: line_coords[2*k+1]=" +
                ::toString(line_coords[2 * k + 1]) + ", expected " +
                ::toString(polygon_coords[j][2 * k + 1]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k + 1] - poly_coords[j][2 * k + 1]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "polygon Y coord test failed: poly_coords[j][2*k+1]=" +
                ::toString(poly_coords[j][2 * k + 1]) + ", expected " +
                ::toString(polygon_coords[j][2 * k + 1]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k + 1] - line[k].y) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "polygon linestring X coord test failed: line[k].y=" +
                ::toString(line[k].y) + ", expected " +
                ::toString(polygon_coords[j][2 * k + 1]) + ".");
          }
        }
      }
    }
  }

  if (count_nulls != 0) {
    return mgr.ERROR_MESSAGE("counting null test failed: count_nulls=" +
                             ::toString(count_nulls) + ", expected 0.");
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
      std::vector<double> line{x[i], y[i], x[i] + dx, y[i] + dy};
      linestrings[i].fromCoords(line);
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t
ct_make_multipolygon__cpu_(TableFunctionManager& mgr,
                           const Column<GeoPolygon>& polygons,
                           Column<GeoMultiPolygon>& mpolygons) {
  auto size = polygons.size();
  mgr.set_output_item_values_total_number(0, polygons.getNofValues());
  mgr.set_output_row_size(size);

  // Initialize mpolygons
  int count_nulls = 0;
  for (int64_t i = 0; i < size; i++) {
    if (polygons.isNull(i)) {
      mpolygons.setNull(i);
      count_nulls++;
    } else {
      std::vector<std::vector<std::vector<double>>> mpolygon_coords;
      mpolygon_coords.reserve(1);
      std::vector<std::vector<double>> polygon_coords = polygons[i].toCoords();
      mpolygon_coords.push_back(polygon_coords);
      auto status = mpolygons[i].fromCoords(mpolygon_coords);
      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
      }
    }
  }

  // Check mpolygons content
  if (count_nulls == 0) {
    return mgr.ERROR_MESSAGE("counting null test failed: count_nulls=" +
                             ::toString(count_nulls) + ", expected non-zero.");
  }

  for (int64_t i = 0; i < size; i++) {
    if (mpolygons.isNull(i)) {
      count_nulls--;
    } else {
      std::vector<std::vector<double>> polygon_coords = polygons[i].toCoords();

      // mpolygons[i] is Geo::MultiPolygon instances
      // mpolygons[i][j] is Geo::Polygon instances
      // mpolygons[i][j][k] is Geo::LineString instances
      // mpolygons[i][j][k][l] is Geo::Point2D instances

      auto nof_polygons = mpolygons[i].size();
      if (nof_polygons != 1) {
        return mgr.ERROR_MESSAGE("multipolygon size test failed: nof_polygons=" +
                                 ::toString(nof_polygons) + ", expected 1.");
      }

      std::vector<std::vector<std::vector<double>>> mpolygon_coords =
          mpolygons[i].toCoords();
      if (nof_polygons != mpolygon_coords.size()) {
        return mgr.ERROR_MESSAGE(
            "multipolygon toCoords size test failed: mpolygon_coords.size()=" +
            ::toString(mpolygon_coords.size()) + ", expected " +
            ::toString(nof_polygons) + ".");
      }

      Geo::Polygon poly = mpolygons[i][0];
      std::vector<std::vector<double>> poly_coords = mpolygon_coords[0];
      if (poly.size() != polygon_coords.size()) {
        return mgr.ERROR_MESSAGE("multipolygon polygon size test failed: poly.size()=" +
                                 ::toString(poly.size()) + ", expected " +
                                 ::toString(polygon_coords.size()) + ".");
      }

      if (poly_coords.size() != polygon_coords.size()) {
        return mgr.ERROR_MESSAGE(
            "multipolygon polygon coords size test failed: poly_coords.size()=" +
            ::toString(poly_coords.size()) + ", expected " +
            ::toString(polygon_coords.size()) + ".");
      }

      for (size_t j = 0; j < poly.size(); j++) {
        Geo::LineString line = poly[j];
        std::vector<double> line_coords = line.toCoords();
        auto nof_points = polygon_coords[j].size() / 2;
        if (poly.size(j) != nof_points) {
          return mgr.ERROR_MESSAGE(
              "multipolygon polygon linestring size test failed: poly.size(" +
              ::toString(j) + ")=" + ::toString(poly.size(j)) + ", expected " +
              ::toString(nof_points) + ".");
        }
        if (line.size() != nof_points) {
          return mgr.ERROR_MESSAGE(
              "multipolygon polygon linestring size test failed: line.size()=" +
              ::toString(line.size()) + ", expected " + ::toString(nof_points) + ".");
        }
        if (poly_coords[j].size() != nof_points * 2) {
          return mgr.ERROR_MESSAGE(
              "multipolygon polygon linestring coords size test failed: "
              "poly_coords[j].size()=" +
              ::toString(poly_coords[j].size()) + ", expected " +
              ::toString(nof_points * 2) + ".");
        }
        if (line_coords.size() != nof_points * 2) {
          return mgr.ERROR_MESSAGE(
              "multipolygon polygon linestring coords size test failed: "
              "line_coords.size()=" +
              ::toString(line_coords.size()) + ", expected " +
              ::toString(nof_points * 2) + ".");
        }

        for (size_t k = 0; k < nof_points; k++) {
          if (std::abs(polygon_coords[j][2 * k] - line_coords[2 * k]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "multipolygon polygon linestring X coord test failed: line_coords[2*k]=" +
                ::toString(line_coords[2 * k]) + ", expected " +
                ::toString(polygon_coords[j][2 * k]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k] - poly_coords[j][2 * k]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "multipolygon polygon X coord test failed: poly_coords[j][2*k]=" +
                ::toString(poly_coords[j][2 * k]) + ", expected " +
                ::toString(polygon_coords[j][2 * k]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k] - line[k].x) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "multipolygon polygon linestring X coord test failed: line[k].x=" +
                ::toString(line[k].x) + ", expected " +
                ::toString(polygon_coords[j][2 * k]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k + 1] - line_coords[2 * k + 1]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "multipolygon polygon linestring Y coord test failed: "
                "line_coords[2*k+1]=" +
                ::toString(line_coords[2 * k + 1]) + ", expected " +
                ::toString(polygon_coords[j][2 * k + 1]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k + 1] - poly_coords[j][2 * k + 1]) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "multipolygon polygon Y coord test failed: poly_coords[j][2*k+1]=" +
                ::toString(poly_coords[j][2 * k + 1]) + ", expected " +
                ::toString(polygon_coords[j][2 * k + 1]) + ".");
          }
          if (std::abs(polygon_coords[j][2 * k + 1] - line[k].y) > 1e-7) {
            return mgr.ERROR_MESSAGE(
                "multipolygon polygon linestring X coord test failed: line[k].y=" +
                ::toString(line[k].y) + ", expected " +
                ::toString(polygon_coords[j][2 * k + 1]) + ".");
          }
        }
      }
    }
  }

  if (count_nulls != 0) {
    return mgr.ERROR_MESSAGE("counting null test failed: count_nulls=" +
                             ::toString(count_nulls) + ", expected 0.");
  }

  return size;
}

EXTENSION_NOINLINE int32_t ct_polygonn__cpu_(TableFunctionManager& mgr,
                                             const Column<GeoMultiPolygon>& mpolygons,
                                             int64_t n,
                                             Column<GeoPolygon>& polygons) {
  auto size = mpolygons.size();
  mgr.set_output_item_values_total_number(0, mpolygons.getNofValues());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (mpolygons.isNull(i)) {
      polygons.setNull(i);
    } else {
      polygons.setItem(i, mpolygons[i][n - 1]);
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t
ct_to_multilinestring__cpu_(TableFunctionManager& mgr,
                            const Column<GeoPolygon>& polygons,
                            Column<GeoMultiLineString>& mlinestrings) {
  auto size = polygons.size();
  mgr.set_output_item_values_total_number(0, polygons.getNofValues());
  mgr.set_output_row_size(size);
  // Initialize mlinestrings
  int count_nulls = 0;
  FlatBufferManager::Status status{};
  for (int64_t i = 0; i < size; i++) {
    if (polygons.isNull(i)) {
      mlinestrings.setNull(i);
      count_nulls++;
    } else {
      std::vector<std::vector<double>> polygon_coords = polygons[i].toCoords();
      status = mlinestrings[i].fromCoords(polygon_coords);
      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
      }
    }
  }
  return size;
}

EXTENSION_NOINLINE int32_t
ct_to_polygon__cpu_(TableFunctionManager& mgr,
                    const Column<GeoMultiLineString>& mlinestrings,
                    Column<GeoPolygon>& polygons) {
  auto size = mlinestrings.size();
  mgr.set_output_item_values_total_number(0, mlinestrings.getNofValues());
  mgr.set_output_row_size(size);
  // Initialize polygons
  int count_nulls = 0;
  FlatBufferManager::Status status{};
  for (int64_t i = 0; i < size; i++) {
    if (mlinestrings.isNull(i)) {
      polygons.setNull(i);
      count_nulls++;
    } else {
      std::vector<std::vector<double>> polygon_coords;
      status = mlinestrings[i].toCoords(polygon_coords);
      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("toCoords failed: " + ::toString(status));
      }
      status = polygons[i].fromCoords(polygon_coords);
      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
      }
    }
  }
  return size;
}

#endif  // #ifndef __CUDACC__
