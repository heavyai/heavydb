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

#include <cmath>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include "Shared/TableFunctionsCommon.hpp"
#include "Shared/math_consts.h"

const size_t max_inputs_per_thread = 1000000L;
const size_t max_temp_output_entries = 200000000L;

// Allow input types to GeoRaster that are different than class types/output Z type
// So we can move everything to the type of T and Z (which can each be either float
// or double)
template <typename T, typename Z>
template <typename T2, typename Z2>
GeoRaster<T, Z>::GeoRaster(const Column<T2>& input_x,
                           const Column<T2>& input_y,
                           const Column<Z2>& input_z,
                           const RasterAggType raster_agg_type,
                           const double bin_dim_meters,
                           const bool geographic_coords,
                           const bool align_bins_to_zero_based_grid)
    : raster_agg_type_(raster_agg_type)
    , bin_dim_meters_(bin_dim_meters)
    , geographic_coords_(geographic_coords)
    , null_sentinel_(inline_null_value<Z>()) {
  auto timer = DEBUG_TIMER(__func__);
  const int64_t input_size{input_z.size()};
  if (input_size <= 0) {
    num_bins_ = 0;
    num_x_bins_ = 0;
    num_y_bins_ = 0;
    return;
  }
  const auto min_max_x = get_column_min_max(input_x);
  const auto min_max_y = get_column_min_max(input_y);
  x_min_ = min_max_x.first;
  x_max_ = min_max_x.second;
  y_min_ = min_max_y.first;
  y_max_ = min_max_y.second;

  if (align_bins_to_zero_based_grid && !geographic_coords_) {
    // For implicit, data-defined bounds, we treat the max of the x and y ranges as
    // inclusive (closed interval), since if the max of the data in either x/y dimensions
    // is at the first value of the next bin, values at that max will be discarded if we
    // don't include the final bin. For exmaple, if the input data (perhaps already binned
    // with a group by query) goes from 0.0 to 40.0 in both x and y directions, we should
    // have the last x/y bins cover the range [40.0, 50.0), not [30.0, 40.0)
    align_bins_max_inclusive();
  }

  calculate_bins_and_scales();
  compute(input_x, input_y, input_z, max_inputs_per_thread);
}

// Allow input types to GeoRaster that are different than class types/output Z type
// So we can move everything to the type of T and Z (which can each be either float
// or double)
template <typename T, typename Z>
template <typename T2, typename Z2>
GeoRaster<T, Z>::GeoRaster(const Column<T2>& input_x,
                           const Column<T2>& input_y,
                           const Column<Z2>& input_z,
                           const RasterAggType raster_agg_type,
                           const double bin_dim_meters,
                           const bool geographic_coords,
                           const bool align_bins_to_zero_based_grid,
                           const T x_min,
                           const T x_max,
                           const T y_min,
                           const T y_max)
    : raster_agg_type_(raster_agg_type)
    , bin_dim_meters_(bin_dim_meters)
    , geographic_coords_(geographic_coords)
    , null_sentinel_(inline_null_value<Z>())
    , x_min_(x_min)
    , x_max_(x_max)
    , y_min_(y_min)
    , y_max_(y_max) {
  auto timer = DEBUG_TIMER(__func__);
  if (align_bins_to_zero_based_grid && !geographic_coords_) {
    // For explicit, user-defined bounds, we treat the max of the x and y ranges as
    // exclusive (open interval), since if the user specifies the max x/y as the end of
    // the bin, they do not intend to add the next full bin For example, if a user
    // specifies a bin_dim_meters of 10.0 and an x and y range from 0 to 40.0, they almost
    // assuredly intend for there to be 4 bins in each of the x and y dimensions, with the
    // last bin of range [30.0, 40.0), not 5 with the final bin's range from [40.0, 50.0)
    align_bins_max_exclusive();
  }
  calculate_bins_and_scales();
  compute(input_x, input_y, input_z, max_inputs_per_thread);
}

template <typename T, typename Z>
inline Z GeoRaster<T, Z>::offset_source_z_from_raster_z(const int64_t source_x_bin,
                                                        const int64_t source_y_bin,
                                                        const Z source_z_offset) const {
  if (is_bin_out_of_bounds(source_x_bin, source_y_bin)) {
    return null_sentinel_;
  }
  const Z terrain_z = z_[x_y_bin_to_bin_index(source_x_bin, source_y_bin, num_x_bins_)];
  if (terrain_z == null_sentinel_) {
    return terrain_z;
  }
  return terrain_z + source_z_offset;
}

template <typename T, typename Z>
inline Z GeoRaster<T, Z>::fill_bin_from_avg_neighbors(const int64_t x_centroid_bin,
                                                      const int64_t y_centroid_bin,
                                                      const int64_t bins_radius) const {
  T val = 0.0;
  int32_t count = 0;
  for (int64_t y_bin = y_centroid_bin - bins_radius;
       y_bin <= y_centroid_bin + bins_radius;
       y_bin++) {
    for (int64_t x_bin = x_centroid_bin - bins_radius;
         x_bin <= x_centroid_bin + bins_radius;
         x_bin++) {
      if (x_bin >= 0 && x_bin < num_x_bins_ && y_bin >= 0 && y_bin < num_y_bins_) {
        const int64_t bin_idx = x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
        const Z bin_val = z_[bin_idx];
        if (bin_val != null_sentinel_) {
          count++;
          val += bin_val;
        }
      }
    }
  }
  return (count == 0) ? null_sentinel_ : val / count;
}

template <typename T, typename Z>
void GeoRaster<T, Z>::align_bins_max_inclusive() {
  x_min_ = std::floor(x_min_ / bin_dim_meters_) * bin_dim_meters_;
  x_max_ = std::floor(x_max_ / bin_dim_meters_) * bin_dim_meters_ +
           bin_dim_meters_;  // Snap to end of bin
  y_min_ = std::floor(y_min_ / bin_dim_meters_) * bin_dim_meters_;
  y_max_ = std::floor(y_max_ / bin_dim_meters_) * bin_dim_meters_ +
           bin_dim_meters_;  // Snap to end of bin
}

template <typename T, typename Z>
void GeoRaster<T, Z>::align_bins_max_exclusive() {
  x_min_ = std::floor(x_min_ / bin_dim_meters_) * bin_dim_meters_;
  x_max_ = std::ceil(x_max_ / bin_dim_meters_) * bin_dim_meters_;
  y_min_ = std::floor(y_min_ / bin_dim_meters_) * bin_dim_meters_;
  y_max_ = std::ceil(y_max_ / bin_dim_meters_) * bin_dim_meters_;
}

template <typename T, typename Z>
void GeoRaster<T, Z>::calculate_bins_and_scales() {
  x_range_ = x_max_ - x_min_;
  y_range_ = y_max_ - y_min_;
  if (geographic_coords_) {
    const T x_centroid = (x_min_ + x_max_) * 0.5;
    const T y_centroid = (y_min_ + y_max_) * 0.5;
    x_meters_per_degree_ =
        distance_in_meters(x_min_, y_centroid, x_max_, y_centroid) / x_range_;

    y_meters_per_degree_ =
        distance_in_meters(x_centroid, y_min_, x_centroid, y_max_) / y_range_;

    num_x_bins_ = x_range_ * x_meters_per_degree_ / bin_dim_meters_;
    num_y_bins_ = y_range_ * y_meters_per_degree_ / bin_dim_meters_;

    x_scale_input_to_bin_ = x_meters_per_degree_ / bin_dim_meters_;
    y_scale_input_to_bin_ = y_meters_per_degree_ / bin_dim_meters_;
    x_scale_bin_to_input_ = bin_dim_meters_ / x_meters_per_degree_;
    y_scale_bin_to_input_ = bin_dim_meters_ / y_meters_per_degree_;

  } else {
    num_x_bins_ = x_range_ / bin_dim_meters_;
    num_y_bins_ = y_range_ / bin_dim_meters_;

    x_scale_input_to_bin_ = 1.0 / bin_dim_meters_;
    y_scale_input_to_bin_ = 1.0 / bin_dim_meters_;
    x_scale_bin_to_input_ = bin_dim_meters_;
    y_scale_bin_to_input_ = bin_dim_meters_;
  }
  num_bins_ = num_x_bins_ * num_y_bins_;
}

template <typename T, typename Z>
template <RasterAggType AggType, typename T2, typename Z2>
void GeoRaster<T, Z>::computeSerialImpl(const Column<T2>& input_x,
                                        const Column<T2>& input_y,
                                        const Column<Z2>& input_z,
                                        std::vector<Z>& output_z) {
  auto timer = DEBUG_TIMER(__func__);
  const int64_t input_size{input_z.size()};
  const Z agg_sentinel = raster_agg_type_ == RasterAggType::MIN
                             ? std::numeric_limits<Z>::max()
                             : std::numeric_limits<Z>::lowest();
  output_z.resize(num_bins_, agg_sentinel);
  ComputeAgg<AggType> compute_agg;
  for (int64_t sparse_idx = 0; sparse_idx != input_size; ++sparse_idx) {
    const int64_t x_bin = get_x_bin(input_x[sparse_idx]);
    const int64_t y_bin = get_y_bin(input_y[sparse_idx]);
    if (x_bin < 0 || x_bin >= num_x_bins_ || y_bin < 0 || y_bin >= num_y_bins_) {
      continue;
    }
    const int64_t bin_idx = x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
    compute_agg(input_z[sparse_idx], output_z[bin_idx], inline_null_value<Z2>());
  }
  for (int64_t bin_idx = 0; bin_idx != num_bins_; ++bin_idx) {
    if (output_z[bin_idx] == agg_sentinel) {
      output_z[bin_idx] = null_sentinel_;
    }
  }
}

template <typename T, typename Z>
template <RasterAggType AggType>
void GeoRaster<T, Z>::computeParallelReductionAggImpl(
    const std::vector<std::vector<Z>>& z_inputs,
    std::vector<Z>& output_z,
    const Z agg_sentinel) {
  const size_t num_inputs = z_inputs.size();
  output_z.resize(num_bins_, agg_sentinel);

  ComputeAgg<AggType> reduction_agg;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_bins_), [&](const tbb::blocked_range<size_t>& r) {
        const size_t start_idx = r.begin();
        const size_t end_idx = r.end();
        for (size_t bin_idx = start_idx; bin_idx != end_idx; ++bin_idx) {
          for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            reduction_agg(z_inputs[input_idx][bin_idx], output_z[bin_idx], agg_sentinel);
          }
        }
      });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_bins_),
                    [&](const tbb::blocked_range<size_t>& r) {
                      const size_t start_idx = r.begin();
                      const size_t end_idx = r.end();
                      for (size_t bin_idx = start_idx; bin_idx != end_idx; ++bin_idx) {
                        if (output_z[bin_idx] == agg_sentinel) {
                          output_z[bin_idx] = null_sentinel_;
                        }
                      }
                    });
}

template <typename T, typename Z>
template <RasterAggType AggType, typename T2, typename Z2>
void GeoRaster<T, Z>::computeParallelImpl(const Column<T2>& input_x,
                                          const Column<T2>& input_y,
                                          const Column<Z2>& input_z,
                                          std::vector<Z>& output_z,
                                          const size_t max_inputs_per_thread) {
  const size_t input_size = input_z.size();
  const size_t max_thread_count = std::thread::hardware_concurrency();
  const size_t num_threads_by_input_elements =
      std::min(max_thread_count,
               ((input_size + max_inputs_per_thread - 1) / max_inputs_per_thread));
  const size_t num_threads_by_output_size =
      std::min(max_thread_count, ((max_temp_output_entries + num_bins_ - 1) / num_bins_));
  const size_t num_threads =
      std::min(num_threads_by_input_elements, num_threads_by_output_size);
  if (num_threads <= 1) {
    computeSerialImpl<AggType, T2, Z2>(input_x, input_y, input_z, output_z);
    return;
  }
  auto timer = DEBUG_TIMER(__func__);

  std::vector<std::vector<Z>> per_thread_z_outputs(num_threads);
  // Fix
  const Z agg_sentinel = raster_agg_type_ == RasterAggType::MIN
                             ? std::numeric_limits<Z>::max()
                             : std::numeric_limits<Z>::lowest();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_threads),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t t = r.begin(); t != r.end(); ++t) {
                        per_thread_z_outputs[t].resize(num_bins_, agg_sentinel);
                      }
                    });

  ComputeAgg<AggType> compute_agg;
  tbb::task_arena limited_arena(num_threads);
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, input_size),
        [&](const tbb::blocked_range<int64_t>& r) {
          const int64_t start_idx = r.begin();
          const int64_t end_idx = r.end();
          size_t thread_idx = tbb::this_task_arena::current_thread_index();
          std::vector<Z>& this_thread_z_output = per_thread_z_outputs[thread_idx];

          for (int64_t sparse_idx = start_idx; sparse_idx != end_idx; ++sparse_idx) {
            const int64_t x_bin = get_x_bin(input_x[sparse_idx]);
            const int64_t y_bin = get_y_bin(input_y[sparse_idx]);
            if (x_bin < 0 || x_bin >= num_x_bins_ || y_bin < 0 || y_bin >= num_y_bins_) {
              continue;
            }
            const int64_t bin_idx = x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
            compute_agg(input_z[sparse_idx],
                        this_thread_z_output[bin_idx],
                        inline_null_value<Z2>());
          }
        });
  });

  // Reduce
  if constexpr (AggType == RasterAggType::COUNT) {
    // Counts can't be counted, they must be summed
    computeParallelReductionAggImpl<RasterAggType::SUM>(
        per_thread_z_outputs, output_z, agg_sentinel);
  } else {
    computeParallelReductionAggImpl<AggType>(
        per_thread_z_outputs, output_z, agg_sentinel);
  }
}

template <typename T, typename Z>
template <typename T2, typename Z2>
void GeoRaster<T, Z>::compute(const Column<T2>& input_x,
                              const Column<T2>& input_y,
                              const Column<Z2>& input_z,
                              const size_t max_inputs_per_thread) {
  switch (raster_agg_type_) {
    case RasterAggType::COUNT: {
      computeParallelImpl<RasterAggType::COUNT, T2, Z2>(
          input_x, input_y, input_z, z_, max_inputs_per_thread);
      break;
    }
    case RasterAggType::MIN: {
      computeParallelImpl<RasterAggType::MIN, T2, Z2>(
          input_x, input_y, input_z, z_, max_inputs_per_thread);
      break;
    }
    case RasterAggType::MAX: {
      computeParallelImpl<RasterAggType::MAX, T2, Z2>(
          input_x, input_y, input_z, z_, max_inputs_per_thread);
      break;
    }
    case RasterAggType::SUM: {
      computeParallelImpl<RasterAggType::SUM, T2, Z2>(
          input_x, input_y, input_z, z_, max_inputs_per_thread);
      break;
    }
    case RasterAggType::AVG: {
      computeParallelImpl<RasterAggType::SUM, T2, Z2>(
          input_x, input_y, input_z, z_, max_inputs_per_thread);
      computeParallelImpl<RasterAggType::COUNT, T2, Z2>(
          input_x, input_y, input_z, counts_, max_inputs_per_thread);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_bins_),
                        [&](const tbb::blocked_range<size_t>& r) {
                          const size_t start_idx = r.begin();
                          const size_t end_idx = r.end();
                          for (size_t bin_idx = start_idx; bin_idx != end_idx;
                               ++bin_idx) {
                            // counts[bin_idx] > 1 will avoid division for nulls and 1
                            // counts
                            if (counts_[bin_idx] > 1) {
                              z_[bin_idx] /= counts_[bin_idx];
                            }
                          }
                        });
      break;
    }
    default: {
      CHECK(false);
    }
  }
}

template <typename T, typename Z>
void GeoRaster<T, Z>::fill_bins_from_neighbors(const int64_t neighborhood_fill_radius,
                                               const bool fill_only_nulls) {
  auto timer = DEBUG_TIMER(__func__);
  std::vector<Z> new_z(num_bins_);
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_y_bins_),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      for (int64_t y_bin = r.begin(); y_bin != r.end(); ++y_bin) {
                        for (int64_t x_bin = 0; x_bin < num_x_bins_; ++x_bin) {
                          const int64_t bin_idx =
                              x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
                          const Z z_val = z_[bin_idx];
                          if (!fill_only_nulls || z_val == null_sentinel_) {
                            new_z[bin_idx] = fill_bin_from_avg_neighbors(
                                x_bin, y_bin, neighborhood_fill_radius);
                          } else {
                            new_z[bin_idx] = z_val;
                          }
                        }
                      }
                    });
  z_.swap(new_z);
}

template <typename T, typename Z>
bool GeoRaster<T, Z>::get_nxn_neighbors_if_not_null(
    const int64_t x_bin,
    const int64_t y_bin,
    const int64_t num_bins_radius,
    std::vector<Z>& neighboring_bins) const {
  const size_t num_bins_per_dim = num_bins_radius * 2 + 1;
  CHECK_EQ(neighboring_bins.size(), num_bins_per_dim * num_bins_per_dim);
  const int64_t end_y_bin_idx = y_bin + num_bins_radius;
  const int64_t end_x_bin_idx = x_bin + num_bins_radius;
  size_t output_bin = 0;
  for (int64_t y_bin_idx = y_bin - num_bins_radius; y_bin_idx <= end_y_bin_idx;
       ++y_bin_idx) {
    for (int64_t x_bin_idx = x_bin - num_bins_radius; x_bin_idx <= end_x_bin_idx;
         ++x_bin_idx) {
      if (x_bin_idx < 0 || x_bin_idx >= num_x_bins_ || y_bin_idx < 0 ||
          y_bin_idx >= num_y_bins_) {
        return false;
      }
      const int64_t bin_idx = x_y_bin_to_bin_index(x_bin_idx, y_bin_idx, num_x_bins_);
      neighboring_bins[output_bin++] = z_[bin_idx];
      if (z_[bin_idx] == null_sentinel_) {
        return false;
      }
    }
  }
  return true;  // not_null
}

template <typename T, typename Z>
inline std::pair<Z, Z> GeoRaster<T, Z>::calculate_slope_and_aspect_of_cell(
    const std::vector<Z>& neighboring_cells,
    const bool compute_slope_in_degrees) const {
  const Z dz_dx =
      ((neighboring_cells[8] + 2 * neighboring_cells[5] + neighboring_cells[2]) -
       (neighboring_cells[6] + 2 * neighboring_cells[3] + neighboring_cells[0])) /
      (8 * bin_dim_meters_);
  const Z dz_dy =
      ((neighboring_cells[6] + 2 * neighboring_cells[7] + neighboring_cells[8]) -
       (neighboring_cells[0] + 2 * neighboring_cells[1] + neighboring_cells[2])) /
      (8 * bin_dim_meters_);
  const Z slope = sqrt(dz_dx * dz_dx + dz_dy * dz_dy);
  std::pair<Z, Z> slope_and_aspect;
  slope_and_aspect.first =
      compute_slope_in_degrees ? atan(slope) * math_consts::radians_to_degrees : slope;
  if (slope < 0.0001) {
    slope_and_aspect.second = null_sentinel_;
  } else {
    const Z aspect_degrees =
        math_consts::radians_to_degrees * atan2(dz_dx, dz_dy);  // -180.0 to 180.0
    slope_and_aspect.second = aspect_degrees + 180.0;
    // aspect_degrees < 0.0 ? 180.0 + aspect_degrees : aspect_degrees;
  }
  return slope_and_aspect;
}

template <typename T, typename Z>
void GeoRaster<T, Z>::calculate_slope_and_aspect(
    Column<Z>& slope,
    Column<Z>& aspect,
    const bool compute_slope_in_degrees) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_EQ(slope.size(), num_bins_);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, num_y_bins_),
      [&](const tbb::blocked_range<int64_t>& r) {
        std::vector<Z> neighboring_z_vals(9);  // 3X3 calc
        for (int64_t y_bin = r.begin(); y_bin != r.end(); ++y_bin) {
          for (int64_t x_bin = 0; x_bin < num_x_bins_; ++x_bin) {
            const bool not_null =
                get_nxn_neighbors_if_not_null(x_bin, y_bin, 1, neighboring_z_vals);
            const int64_t bin_idx = x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
            if (!not_null) {
              slope.setNull(bin_idx);
              aspect.setNull(bin_idx);
            } else {
              const auto slope_and_aspect = calculate_slope_and_aspect_of_cell(
                  neighboring_z_vals, compute_slope_in_degrees);
              slope[bin_idx] = slope_and_aspect.first;
              if (slope_and_aspect.second == null_sentinel_) {
                aspect.setNull(bin_idx);
              } else {
                aspect[bin_idx] = slope_and_aspect.second;
              }
            }
          }
        }
      });
}

template <typename T, typename Z>
int64_t GeoRaster<T, Z>::outputDenseColumns(
    TableFunctionManager& mgr,
    Column<T>& output_x,
    Column<T>& output_y,
    Column<Z>& output_z,
    const int64_t neighborhood_null_fill_radius) const {
  auto timer = DEBUG_TIMER(__func__);
  mgr.set_output_row_size(num_bins_);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, num_y_bins_),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t y_bin = r.begin(); y_bin != r.end(); ++y_bin) {
          for (int64_t x_bin = 0; x_bin < num_x_bins_; ++x_bin) {
            const int64_t bin_idx = x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
            output_x[bin_idx] = x_min_ + (x_bin + 0.5) * x_scale_bin_to_input_;
            output_y[bin_idx] = y_min_ + (y_bin + 0.5) * y_scale_bin_to_input_;
            const Z z_val = z_[bin_idx];
            if (z_val == null_sentinel_) {
              output_z.setNull(bin_idx);
              if (neighborhood_null_fill_radius) {
                const Z avg_neighbor_value = fill_bin_from_avg_neighbors(
                    x_bin, y_bin, neighborhood_null_fill_radius);
                if (avg_neighbor_value != null_sentinel_) {
                  output_z[bin_idx] = avg_neighbor_value;
                }
              }
            } else {
              output_z[bin_idx] = z_[bin_idx];
            }
          }
        }
      });
  return num_bins_;
}

template <typename T, typename Z>
int64_t GeoRaster<T, Z>::outputDenseColumns(TableFunctionManager& mgr,
                                            Column<T>& output_x,
                                            Column<T>& output_y,
                                            Column<Z>& output_z) const {
  auto timer = DEBUG_TIMER(__func__);
  mgr.set_output_row_size(num_bins_);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, num_y_bins_),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t y_bin = r.begin(); y_bin != r.end(); ++y_bin) {
          for (int64_t x_bin = 0; x_bin < num_x_bins_; ++x_bin) {
            const int64_t bin_idx = x_y_bin_to_bin_index(x_bin, y_bin, num_x_bins_);
            output_x[bin_idx] = x_min_ + (x_bin + 0.5) * x_scale_bin_to_input_;
            output_y[bin_idx] = y_min_ + (y_bin + 0.5) * y_scale_bin_to_input_;
            const Z z_val = z_[bin_idx];
            if (z_val == null_sentinel_) {
              output_z.setNull(bin_idx);
            } else {
              output_z[bin_idx] = z_[bin_idx];
            }
          }
        }
      });
  return num_bins_;
}

#endif  // __CUDACC__
