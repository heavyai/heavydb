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

#pragma once

#ifndef __CUDACC__

#include <vector>
#include "QueryEngine/heavydbTypes.h"

enum class RasterAggType { COUNT, MIN, MAX, SUM, AVG, INVALID };

RasterAggType get_raster_agg_type(const std::string& agg_type_str) {
  const auto upper_agg_type_str = to_upper(agg_type_str);
  const static std::map<std::string, RasterAggType> agg_type_map = {
      {"COUNT", RasterAggType::COUNT},
      {"MIN", RasterAggType::MIN},
      {"MAX", RasterAggType::MAX},
      {"SUM", RasterAggType::SUM},
      {"AVG", RasterAggType::AVG}};
  const auto itr = agg_type_map.find(upper_agg_type_str);
  if (itr == agg_type_map.end()) {
    return RasterAggType::INVALID;
  }
  return itr->second;
}

template <RasterAggType AggType>
struct ComputeAgg {};

template <>
struct ComputeAgg<RasterAggType::COUNT> {
  template <typename Z, typename Z2>
  void operator()(const Z2& input_z, Z& output_z, const Z2 input_sentinel) {
    if (input_z != input_sentinel) {
      output_z = output_z == std::numeric_limits<Z>::lowest() ? 1 : output_z + 1;
    }
  }
};

template <>
struct ComputeAgg<RasterAggType::MAX> {
  template <typename Z, typename Z2>
  void operator()(const Z2& input_z, Z& output_z, const Z2 input_sentinel) {
    if (input_z != input_sentinel && input_z > output_z) {
      output_z = input_z;
    }
  }
};

template <>
struct ComputeAgg<RasterAggType::MIN> {
  template <typename Z, typename Z2>
  void operator()(const Z2& input_z, Z& output_z, const Z2 input_sentinel) {
    if (input_z != input_sentinel && input_z < output_z) {
      output_z = input_z;
    }
  }
};

template <>
struct ComputeAgg<RasterAggType::SUM> {
  template <typename Z, typename Z2>
  void operator()(const Z2& input_z, Z& output_z, const Z2 input_sentinel) {
    if (input_z != input_sentinel) {
      output_z =
          output_z == std::numeric_limits<Z>::lowest() ? input_z : output_z + input_z;
    }
  }
};

template <typename T, typename Z>
struct GeoRaster {
  const RasterAggType raster_agg_type_;
  const T bin_dim_meters_;
  const bool geographic_coords_;
  const Z null_sentinel_;
  std::vector<Z> z_;
  std::vector<Z> counts_;
  T x_min_{0};
  T x_max_{0};
  T y_min_{0};
  T y_max_{0};
  T x_range_{0};
  T y_range_{0};
  T x_meters_per_degree_{0};
  T y_meters_per_degree_{0};
  int64_t num_x_bins_{0};
  int64_t num_y_bins_{0};
  int64_t num_bins_{0};
  T x_scale_input_to_bin_{0};
  T y_scale_input_to_bin_{0};
  T x_scale_bin_to_input_{0};
  T y_scale_bin_to_input_{0};

  template <typename T2, typename Z2>
  GeoRaster(const Column<T2>& input_x,
            const Column<T2>& input_y,
            const Column<Z2>& input_z,
            const RasterAggType raster_agg_type,
            const double bin_dim_meters,
            const bool geographic_coords,
            const bool align_bins_to_zero_based_grid);

  template <typename T2, typename Z2>
  GeoRaster(const Column<T2>& input_x,
            const Column<T2>& input_y,
            const Column<Z2>& input_z,
            const RasterAggType raster_agg_type,
            const double bin_dim_meters,
            const bool geographic_coords,
            const bool align_bins_to_zero_based_grid,
            const T x_min,
            const T x_max,
            const T y_min,
            const T y_max);

  inline T get_x_bin(const T input) const {
    return (input - x_min_) * x_scale_input_to_bin_;
  }

  inline T get_y_bin(const T input) const {
    return (input - y_min_) * y_scale_input_to_bin_;
  }

  inline bool is_null(const Z value) const { return value == null_sentinel_; }

  inline bool is_bin_out_of_bounds(const int64_t source_x_bin,
                                   const int64_t source_y_bin) const {
    return (source_x_bin < 0 || source_x_bin >= num_x_bins_ || source_y_bin < 0 ||
            source_y_bin >= num_y_bins_);
  }

  inline Z offset_source_z_from_raster_z(const int64_t source_x_bin,
                                         const int64_t source_y_bin,
                                         const Z source_z_offset) const;

  inline Z fill_bin_from_avg_neighbors(const int64_t x_centroid_bin,
                                       const int64_t y_centroid_bin,
                                       const int64_t bins_radius) const;

  void align_bins_max_inclusive();

  void align_bins_max_exclusive();

  void calculate_bins_and_scales();

  template <typename T2, typename Z2>
  void compute(const Column<T2>& input_x,
               const Column<T2>& input_y,
               const Column<Z2>& input_z,
               const size_t max_inputs_per_thread);

  void fill_bins_from_neighbors(const int64_t neighborhood_fill_radius,
                                const bool fill_only_nulls);

  bool get_nxn_neighbors_if_not_null(const int64_t x_bin,
                                     const int64_t y_bin,
                                     const int64_t num_bins_radius,
                                     std::vector<Z>& neighboring_bins) const;
  inline std::pair<Z, Z> calculate_slope_and_aspect_of_cell(
      const std::vector<Z>& neighboring_cells,
      const bool compute_slope_in_degrees) const;
  void calculate_slope_and_aspect(Column<Z>& slope,
                                  Column<Z>& aspect,
                                  const bool compute_slope_in_degrees) const;

  int64_t outputDenseColumns(TableFunctionManager& mgr,
                             Column<T>& output_x,
                             Column<T>& output_y,
                             Column<Z>& output_z) const;

  int64_t outputDenseColumns(TableFunctionManager& mgr,
                             Column<T>& output_x,
                             Column<T>& output_y,
                             Column<Z>& output_z,
                             const int64_t neighborhood_null_fill_radius) const;

  void setMetadata(TableFunctionManager& mgr) const;

 private:
  template <RasterAggType AggType, typename T2, typename Z2>
  void computeSerialImpl(const Column<T2>& input_x,
                         const Column<T2>& input_y,
                         const Column<Z2>& input_z,
                         std::vector<Z>& output_z);

  template <RasterAggType AggType>
  void computeParallelReductionAggImpl(const std::vector<std::vector<Z>>& z_inputs,
                                       std::vector<Z>& output_z,
                                       const Z agg_sentinel);

  template <RasterAggType AggType, typename T2, typename Z2>
  void computeParallelImpl(const Column<T2>& input_x,
                           const Column<T2>& input_y,
                           const Column<Z2>& input_z,
                           std::vector<Z>& output_z,
                           const size_t max_inputs_per_thread);
};

// clang-format off
/*
  UDTF: tf_geo_rasterize__cpu_template(TableFunctionManager,
  Cursor<Column<T> x, Column<T> y, Column<Z> z> raster, TextEncodingNone agg_type,
  T bin_dim_meters | require="bin_dim_meters > 0", bool geographic_coords,
  int64_t neighborhood_fill_radius | require="neighborhood_fill_radius >= 0",
  bool fill_only_nulls) | filter_table_function_transpose=on ->
  Column<T> x, Column<T> y, Column<Z> z, T=[float, double], Z=[float, double]
 */
// clang-format on

template <typename T, typename Z>
TEMPLATE_NOINLINE int32_t
tf_geo_rasterize__cpu_template(TableFunctionManager& mgr,
                               const Column<T>& input_x,
                               const Column<T>& input_y,
                               const Column<Z>& input_z,
                               const TextEncodingNone& agg_type_str,
                               const T bin_dim_meters,
                               const bool geographic_coords,
                               const int64_t neighborhood_fill_radius,
                               const bool fill_only_nulls,
                               Column<T>& output_x,
                               Column<T>& output_y,
                               Column<Z>& output_z) {
  const auto raster_agg_type = get_raster_agg_type(agg_type_str);
  if (raster_agg_type == RasterAggType::INVALID) {
    const std::string error_msg =
        "Invalid Raster Aggregate Type: " + std::string(agg_type_str);
    return mgr.ERROR_MESSAGE(error_msg);
  }
  GeoRaster<T, Z> geo_raster(input_x,
                             input_y,
                             input_z,
                             raster_agg_type,
                             bin_dim_meters,
                             geographic_coords,
                             true);

  geo_raster.setMetadata(mgr);

  if (neighborhood_fill_radius > 0) {
    geo_raster.fill_bins_from_neighbors(neighborhood_fill_radius, fill_only_nulls);
  }

  return geo_raster.outputDenseColumns(mgr, output_x, output_y, output_z);
}

// clang-format off
/*
  UDTF: tf_geo_rasterize__cpu_template(TableFunctionManager,
  Cursor<Column<T> x, Column<T> y, Column<Z> z> raster,
  TextEncodingNone agg_type, T bin_dim_meters | require="bin_dim_meters > 0",
  bool geographic_coords, int64_t neighborhood_fill_radius | require="neighborhood_fill_radius >= 0", bool fill_only_nulls,
  T x_min, T x_max | require="x_max > x_min", T y_min, T y_max | require="y_max > y_min") ->
  Column<T> x, Column<T> y, Column<Z> z, T=[float, double], Z=[float, double]
 */
// clang-format on

template <typename T, typename Z>
TEMPLATE_NOINLINE int32_t
tf_geo_rasterize__cpu_template(TableFunctionManager& mgr,
                               const Column<T>& input_x,
                               const Column<T>& input_y,
                               const Column<Z>& input_z,
                               const TextEncodingNone& agg_type_str,
                               const T bin_dim_meters,
                               const bool geographic_coords,
                               const int64_t neighborhood_fill_radius,
                               const bool fill_only_nulls,
                               const T x_min,
                               const T x_max,
                               const T y_min,
                               const T y_max,
                               Column<T>& output_x,
                               Column<T>& output_y,
                               Column<Z>& output_z) {
  const auto raster_agg_type = get_raster_agg_type(agg_type_str);
  if (raster_agg_type == RasterAggType::INVALID) {
    const std::string error_msg =
        "Invalid Raster Aggregate Type: " + std::string(agg_type_str);
    return mgr.ERROR_MESSAGE(error_msg);
  }

  GeoRaster<T, Z> geo_raster(input_x,
                             input_y,
                             input_z,
                             raster_agg_type,
                             bin_dim_meters,
                             geographic_coords,
                             true,
                             x_min,
                             x_max,
                             y_min,
                             y_max);

  geo_raster.setMetadata(mgr);

  if (neighborhood_fill_radius > 0) {
    geo_raster.fill_bins_from_neighbors(neighborhood_fill_radius, fill_only_nulls);
  }

  return geo_raster.outputDenseColumns(mgr, output_x, output_y, output_z);
}
// clang-format off
/*
  UDTF: tf_geo_rasterize_slope__cpu_template(TableFunctionManager,
  Cursor<Column<T> x, Column<T> y, Column<Z> z> raster,
  TextEncodingNone agg_type, T bin_dim_meters | require="bin_dim_meters > 0",
  bool geographic_coords, int64_t neighborhood_fill_radius | require="neighborhood_fill_radius >= 0", bool fill_only_nulls,
  bool compute_slope_in_degrees) | filter_table_function_transpose=on ->
  Column<T> x, Column<T> y, Column<Z> z, Column<Z> slope, Column<Z> aspect, T=[float, double], Z=[float, double]
 */
// clang-format on

template <typename T, typename Z>
TEMPLATE_NOINLINE int32_t
tf_geo_rasterize_slope__cpu_template(TableFunctionManager& mgr,
                                     const Column<T>& input_x,
                                     const Column<T>& input_y,
                                     const Column<Z>& input_z,
                                     const TextEncodingNone& agg_type_str,
                                     const T bin_dim_meters,
                                     const bool geographic_coords,
                                     const int64_t neighborhood_fill_radius,
                                     const bool fill_only_nulls,
                                     const bool compute_slope_in_degrees,
                                     Column<T>& output_x,
                                     Column<T>& output_y,
                                     Column<Z>& output_z,
                                     Column<Z>& output_slope,
                                     Column<Z>& output_aspect) {
  const auto raster_agg_type = get_raster_agg_type(agg_type_str);
  if (raster_agg_type == RasterAggType::INVALID) {
    const std::string error_msg =
        "Invalid Raster Aggregate Type: " + std::string(agg_type_str);
    return mgr.ERROR_MESSAGE(error_msg);
  }

  GeoRaster<T, Z> geo_raster(input_x,
                             input_y,
                             input_z,
                             raster_agg_type,
                             bin_dim_meters,
                             geographic_coords,
                             true);
  geo_raster.setMetadata(mgr);

  if (neighborhood_fill_radius > 0) {
    geo_raster.fill_bins_from_neighbors(neighborhood_fill_radius, fill_only_nulls);
  }

  const size_t output_rows =
      geo_raster.outputDenseColumns(mgr, output_x, output_y, output_z);
  geo_raster.calculate_slope_and_aspect(
      output_slope, output_aspect, compute_slope_in_degrees);
  return output_rows;
}

#include "GeoRasterTableFunctions.cpp"

#endif  // __CUDACC__
