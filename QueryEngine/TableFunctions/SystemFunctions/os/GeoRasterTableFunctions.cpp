/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "GeoRasterTableFunctions.hpp"

RasterAggType get_raster_agg_type(const std::string& agg_type_str,
                                  const bool is_fill_agg) {
  const auto upper_agg_type_str = to_upper(agg_type_str);
  const static std::map<std::string, RasterAggType> agg_type_map = {
      {"COUNT", RasterAggType::COUNT},
      {"MIN", RasterAggType::MIN},
      {"MAX", RasterAggType::MAX},
      {"SUM", RasterAggType::SUM},
      {"AVG", RasterAggType::AVG},
      {"GAUSS_AVG", RasterAggType::GAUSS_AVG},
      {"BOX_AVG", RasterAggType::BOX_AVG}};
  const auto itr = agg_type_map.find(upper_agg_type_str);
  if (itr == agg_type_map.end()) {
    return RasterAggType::INVALID;
  }
  if (is_fill_agg && itr->second == RasterAggType::AVG) {
    return RasterAggType::GAUSS_AVG;
  } else if (!is_fill_agg && (itr->second == RasterAggType::BOX_AVG ||
                              itr->second == RasterAggType::GAUSS_AVG)) {
    // GAUSS_AVG and BOX_AVG are fill-only aggregates
    return RasterAggType::INVALID;
  }
  return itr->second;
}

std::vector<double> generate_1d_gaussian_kernel(const int64_t fill_radius, double sigma) {
  const int64_t kernel_size = fill_radius * 2 + 1;
  std::vector<double> gaussian_kernel(kernel_size);
  const double expr = 1.0 / (sigma * sqrt(2.0 * M_PI));
  for (int64_t kernel_idx = -fill_radius; kernel_idx <= fill_radius; ++kernel_idx) {
    gaussian_kernel[kernel_idx + fill_radius] =
        expr * exp((kernel_idx * kernel_idx) / (-2.0 * (sigma * sigma)));
  }
  return gaussian_kernel;
}

#endif  // __CUDACC__
