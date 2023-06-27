/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#pragma once

#ifndef __CUDACC__

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"
#include "QueryEngine/heavydbTypes.h"
#include "Shared/measure.h"
#include "ThirdParty/kdtree-cpp/kdtree.hpp"

namespace CrossSectionTableFunctions {

//
// 1D implementation
//

template <typename TLL, typename TV>
int32_t tf_cross_section_1d_impl(TableFunctionManager& mgr,
                                 const Column<TLL>& lon,
                                 const Column<TLL>& lat,
                                 const Column<TV>& values,
                                 const TLL line_x1,
                                 const TLL line_y1,
                                 const TLL line_x2,
                                 const TLL line_y2,
                                 const int32_t num_points,
                                 Column<GeoLineString>& line) {
  // supported types
  static_assert(std::is_same_v<double, TLL>);
  static_assert(std::is_same_v<float, TV> || std::is_same_v<double, TV>);

  // validate sizes
  CHECK_EQ(lon.size(), lat.size()) << "input column size mismatch";
  CHECK_EQ(lon.size(), values.size()) << "input column size mismatch";

  // enforce parser validation
  CHECK_GT(num_points, 1) << "num_points must be > 1";

  // any raster data?
  if (values.size() == 0) {
    // return LINESTRING EMPTY
    mgr.set_output_array_values_total_number(0, 0);
    mgr.set_output_row_size(1);
    return 1;
  }

  // report values (verbose only)
  VLOG(2) << "tf_cross_section_1d: Input raster data has " << values.size()
          << " values, min "
          << *std::min_element(values.getPtr(), values.getPtr() + values.size())
          << ", max "
          << *std::max_element(values.getPtr(), values.getPtr() + values.size());

  // buckets
  std::vector<TV> bucket_values(num_points, static_cast<TV>(0));
  std::vector<int32_t> bucket_counts(num_points, 0);

  // start timer
  auto scatter_timer = timer_start();

  // scatter points into buckets
  // help from http://www.sunshine2k.de/coding/java/PointOnLine/PointOnLine.html
  const TLL e1x = line_x2 - line_x1;
  const TLL e1y = line_y2 - line_y1;
  const TLL len2_e1 = (e1x * e1x) + (e1y * e1y);
  int32_t bucket_misses = 0;
  for (int i = 0; i < lon.size(); i++) {
    // project point to line
    const TLL e2x = lon[i] - line_x1;
    const TLL e2y = lat[i] - line_y1;
    const TLL dp_e1e2 = (e1x * e2x) + (e1y * e2y);
    const TLL normalized_distance = dp_e1e2 / len2_e1;
    // find bucket
    auto const bucket_index = static_cast<int32_t>(normalized_distance * num_points);
    // add to bucket?
    if (bucket_index >= 0 && bucket_index < num_points) {
      bucket_values[bucket_index] += values[i];
      bucket_counts[bucket_index]++;
    } else {
      bucket_misses++;
    }
  }

  // stop timer
  LOG(INFO) << "tf_cross_section_1d: scatter took " << timer_stop(scatter_timer) << "ms";

  // warn if any misses
  LOG_IF(WARNING, bucket_misses > 0)
      << "tf_cross_section_1d: had " << bucket_misses << " bucket misses";

  // size outputs
  mgr.set_output_array_values_total_number(0, num_points * 2);
  mgr.set_output_row_size(1);

  // generate LINESTRING
  std::vector<double> coords;
  coords.reserve(num_points * 2);
  for (int i = 0; i < num_points; i++) {
    auto const x = static_cast<TV>(i) / static_cast<TV>(num_points - 1);
    auto const y = bucket_counts[i]
                       ? (bucket_values[i] / static_cast<TV>(bucket_counts[i]))
                       : static_cast<TV>(0);
    coords.push_back(x);
    coords.push_back(y);
  }

  // set output linestring
  line[0].fromCoords(coords);

  // done
  return 1;
}

//
// 2D implementation
//

template <typename TLL, typename TY, typename TV>
int32_t tf_cross_section_2d_impl(TableFunctionManager& mgr,
                                 const Column<TLL>& lon,
                                 const Column<TLL>& lat,
                                 const Column<TY>& y_axis,
                                 const Column<TV>& values,
                                 const TLL line_x1,
                                 const TLL line_y1,
                                 const TLL line_x2,
                                 const TLL line_y2,
                                 const int32_t num_points_x,
                                 const int32_t num_points_y,
                                 const TLL dwithin_distance,
                                 Column<TLL>& x,
                                 Column<TLL>& y,
                                 Column<TV>& color) {
  // supported types
  static_assert(std::is_same_v<double, TLL>);
  static_assert(std::is_same_v<float, TY> || std::is_same_v<double, TY>);
  static_assert(std::is_same_v<float, TV> || std::is_same_v<double, TV>);

  // validate sizes
  CHECK_EQ(lon.size(), lat.size()) << "input column size mismatch";
  CHECK_EQ(lon.size(), y_axis.size()) << "input column size mismatch";
  CHECK_EQ(lon.size(), values.size()) << "input column size mismatch";

  // enforce parser validation
  CHECK_GT(num_points_x, 1) << "num_points_x must be > 1";
  CHECK_GT(num_points_y, 1) << "num_points_y must be > 1";

  static constexpr int kNumNearest = 3;

  // enough raster data?
  if (values.size() < kNumNearest) {
    mgr.set_output_item_values_total_number(0, 0);
    mgr.set_output_item_values_total_number(1, 0);
    mgr.set_output_item_values_total_number(2, 0);
    mgr.set_output_row_size(0);
    return 0;
  }

  // report values (verbose only)
  VLOG(2) << "tf_cross_section_2d: Input raster data has " << values.size()
          << " values, min "
          << *std::min_element(values.getPtr(), values.getPtr() + values.size())
          << ", max "
          << *std::max_element(values.getPtr(), values.getPtr() + values.size());

  //
  // build points for tree and capture y-axis range
  //

  auto build_nodes_timer = timer_start();

  auto y_min = std::numeric_limits<double>::max();
  auto y_max = -std::numeric_limits<double>::max();

  Kdtree::KdNodeVector nodes(lon.size());

  for (int i = 0; i < lon.size(); i++) {
    auto& n = nodes[i];
    auto const dlon = static_cast<double>(lon[i]);
    auto const dlat = static_cast<double>(lat[i]);
    auto const dy = static_cast<double>(y_axis[i]);
    n.point = {dlon, dlat, dy};
    n.data = nullptr;
    n.index = i;
    y_min = std::min(dy, y_min);
    y_max = std::max(dy, y_max);
  }

  LOG(INFO) << "tf_cross_section_2d: build nodes took " << timer_stop(build_nodes_timer)
            << "ms";

  //
  // build tree
  //

  auto build_tree_timer = timer_start();

  Kdtree::KdTree tree(&nodes);

  LOG(INFO) << "tf_cross_section_2d: build tree took " << timer_stop(build_tree_timer)
            << "ms";

  //
  // size outputs
  //

  auto const num_output_rows = num_points_x * num_points_y;

  mgr.set_output_item_values_total_number(0, num_output_rows);
  mgr.set_output_item_values_total_number(1, num_output_rows);
  mgr.set_output_item_values_total_number(2, num_output_rows);
  mgr.set_output_row_size(num_output_rows);

  //
  // compute mesh
  //

  auto compute_mesh_timer = timer_start();

  auto const max_dist2 = dwithin_distance * dwithin_distance;

  static constexpr double kMinDistance2 = 0.0001;

  tbb::parallel_for(
      tbb::blocked_range<int32_t>(0, num_output_rows),
      [&](const tbb::blocked_range<int32_t>& r) {
        auto const start_idx = r.begin();
        auto const end_idx = r.end();
        for (int32_t output_index = start_idx; output_index < end_idx; output_index++) {
          // discrete mesh-space point
          auto const xi = output_index % num_points_x;
          auto const yi = output_index / num_points_x;

          // normalized mesh-space point
          auto const tx = static_cast<double>(xi) / static_cast<double>(num_points_x - 1);
          auto const ty = static_cast<double>(yi) / static_cast<double>(num_points_y - 1);

          // world-space point
          std::vector<double> p{(tx * (line_x2 - line_x1)) + line_x1,
                                (tx * (line_y2 - line_y1)) + line_y1,
                                (ty * (y_max - y_min)) + y_min};

          // get nearest N points from tree
          Kdtree::KdNodeVector result;
          tree.k_nearest_neighbors(p, kNumNearest, &result);
          CHECK_EQ(result.size(), kNumNearest);

          // are the points close enough in lon/lat space?
          int valid_indices[kNumNearest];
          double valid_weights[kNumNearest];
          double total_weight = 0.0;
          size_t num_valid = 0u;
          for (size_t i = 0; i < kNumNearest; i++) {
            auto const dx = result[i].point[0] - p[0];
            auto const dy = result[i].point[1] - p[1];
            auto const dist2 = (dx * dx) + (dy * dy);
            // discard points that are more than one mesh grid cell away
            if (dist2 < max_dist2) {
              // weights are inverse squared distance
              auto const dz = result[i].point[2] - p[2];
              auto const len2 = dist2 + (dz * dz);
              auto const weight = 1.0 / std::max(len2, kMinDistance2);
              // use this point
              valid_indices[num_valid] = i;
              valid_weights[num_valid] = weight;
              total_weight += weight;
              num_valid++;
            }
          }

          // compute final weighted value
          TV col;
          if constexpr (std::is_same_v<double, TV>) {
            col = static_cast<TV>(NULL_DOUBLE);
          } else {
            col = static_cast<TV>(NULL_FLOAT);
          }
          if (num_valid > 0u && total_weight > 0.0) {
            col = static_cast<TV>(0.0);
            const double weight_multiplier = 1.0 / total_weight;
            for (size_t i = 0; i < num_valid; i++) {
              auto const index = valid_indices[i];
              auto const weight = valid_weights[i];
              auto const value = values[result[index].index];
              col += static_cast<TV>(value * weight * weight_multiplier);
            }
          }

          // output the screen-space point and color
          x[output_index] = tx;
          y[output_index] = p[2];
          color[output_index] = col;
        }
      });

  LOG(INFO) << "tf_cross_section_2d: compute mesh took " << timer_stop(compute_mesh_timer)
            << "ms";

  // done
  return num_output_rows;
}

}  // namespace CrossSectionTableFunctions

//
// public TFs
//

// clang-format off
/*
  UDTF: tf_cross_section_1d__cpu_template(TableFunctionManager mgr,
  Cursor<Column<TLL> lon, Column<TLL> lat, Column<TV> values> raster,
  TLL line_x1, TLL line_y1, TLL line_x2, TLL line_y2,
  int32_t num_points | require="num_points > 1") -> Column<GeoLineString> line,
  TLL=[double], TV=[float,double]
 */
// clang-format on

template <typename TLL, typename TV>
TEMPLATE_NOINLINE int32_t tf_cross_section_1d__cpu_template(TableFunctionManager& mgr,
                                                            const Column<TLL>& lon,
                                                            const Column<TLL>& lat,
                                                            const Column<TV>& values,
                                                            const TLL line_x1,
                                                            const TLL line_y1,
                                                            const TLL line_x2,
                                                            const TLL line_y2,
                                                            const int32_t num_points,
                                                            Column<GeoLineString>& line) {
  return CrossSectionTableFunctions::tf_cross_section_1d_impl<TLL, TV>(
      mgr, lon, lat, values, line_x1, line_y1, line_x2, line_y2, num_points, line);
}

// clang-format off
/*
  UDTF: tf_cross_section_2d__cpu_template(TableFunctionManager mgr,
  Cursor<Column<TLL> lon, Column<TLL> lat, Column<TY> y_axis, Column<TV> values> raster,
  TLL line_x1, TLL line_y1, TLL line_x2, TLL line_y2,
  int32_t num_points_x | require="num_points_x > 1",
  int32_t num_points_y | require="num_points_y > 1",
  double dwithin_distance | require="dwithin_distance > 0.0") -> Column<TLL> x, Column<TLL> y, Column<TV> color,
  TLL=[double], TY=[float,double], TV=[float,double]
 */
// clang-format on

template <typename TLL, typename TY, typename TV>
TEMPLATE_NOINLINE int32_t tf_cross_section_2d__cpu_template(TableFunctionManager& mgr,
                                                            const Column<TLL>& lon,
                                                            const Column<TLL>& lat,
                                                            const Column<TY>& y_axis,
                                                            const Column<TV>& values,
                                                            const TLL line_x1,
                                                            const TLL line_y1,
                                                            const TLL line_x2,
                                                            const TLL line_y2,
                                                            const int32_t num_points_x,
                                                            const int32_t num_points_y,
                                                            const TLL dwithin_distance,
                                                            Column<TLL>& x,
                                                            Column<TLL>& y,
                                                            Column<TV>& color) {
  return CrossSectionTableFunctions::tf_cross_section_2d_impl<TLL, TY, TV>(
      mgr,
      lon,
      lat,
      y_axis,
      values,
      line_x1,
      line_y1,
      line_x2,
      line_y2,
      num_points_x,
      num_points_y,
      dwithin_distance,
      x,
      y,
      color);
}

#endif  // __CUDACC__
