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
  VLOG(2) << "tf_cross_section: Input raster data has " << values.size()
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
  LOG(INFO) << "tf_cross_section: scatter took " << timer_stop(scatter_timer) << "ms";

  // warn if any misses
  LOG_IF(WARNING, bucket_misses > 0)
      << "tf_cross_section: had " << bucket_misses << " bucket misses";

  // size outputs
  mgr.set_output_array_values_total_number(0, num_points * 2);
  mgr.set_output_row_size(1);

  // generate LINESTRING
  std::vector<double> coords;
  coords.reserve(num_points * 2);
  for (int i = 0; i < num_points; i++) {
    auto const x = static_cast<TV>(i) / static_cast<TV>(num_points);
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

#endif  // __CUDACC__
