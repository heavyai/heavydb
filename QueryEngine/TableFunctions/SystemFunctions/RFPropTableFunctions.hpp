/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef HAVE_RF_PROP_TFS
#ifndef __CUDACC__

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <boost/timer/timer.hpp>

#include "QueryEngine/TableFunctions/SystemFunctions/GeoRasterTableFunctions.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/Shared/TableFunctionsCommon.hpp"
#include "QueryEngine/heavydbTypes.h"

#include "AngleAttenuations.h"
#include "AntennaPattern.h"

constexpr int32_t bin_min_sentinel{std::numeric_limits<int32_t>::lowest()};
constexpr double ray_step_epsilon{0.00001};

template <typename T>
inline double convert_power_watts_to_dbm(const T power_watts) {
  return log10(power_watts) * 10.0 + 30.0;
}

template <typename T>
inline double convert_power_dbm_to_watts(const T power_dbm) {
  return pow(10.0, (power_dbm - 30.0) * .1);
}

template <typename T>
struct GridCell {
  int32_t bin_idx;
  int32_t x_bin_idx;
  int32_t y_bin_idx;
  T xy_distance;
  T inverted_xy_distance;
  T xy_angle;
  int32_t parent_idx;

  void print() const {
    std::cout << "Bin idx: " << bin_idx << std::endl;
    std::cout << "X Bin idx: " << x_bin_idx << std::endl;
    std::cout << "Y Bin idx: " << y_bin_idx << std::endl;
    std::cout << "XY Distance: " << xy_distance << std::endl;
    std::cout << "XY Angle: " << xy_angle << std::endl;
    std::cout << "Parent idx: " << parent_idx << std::endl;
  }
};

template <typename T>
bool sort_by_distance(const GridCell<T>& a, const GridCell<T>& b) {
  return (a.xy_distance < b.xy_distance);
}

template <typename T>
struct RasterOffsets {
  RasterOffsets(const double radius,
                const double bin_dim,
                const bool only_first_quarter_radian = false);

  inline GridCell<T> getGridCell(const int32_t idx) const { return grid_cells_[idx]; }

  inline GridCell<T> getGridCellByBinIdx(const int32_t bin_idx) const {
    const int32_t permuted_idx = distance_bin_permuted_idxs_[bin_idx];
    return grid_cells_[permuted_idx];
  }

  inline GridCell<T> getGridCellByXYBinIdx(const int32_t x_bin_idx,
                                           const int32_t y_bin_idx) const {
    const int32_t bin_idx = x_y_bin_to_bin_index(
        x_bin_idx + bin_radius_, y_bin_idx + bin_radius_, grid_stride_);
    const int32_t permuted_idx = distance_bin_permuted_idxs_[bin_idx];
    return grid_cells_[permuted_idx];
  }

  inline int32_t getPermutedGridIdx(const int32_t x_bin_idx,
                                    const int32_t y_bin_idx) const {
    const int32_t bin_idx = x_y_bin_to_bin_index(
        x_bin_idx + bin_radius_, y_bin_idx + bin_radius_, grid_stride_);
    return distance_bin_permuted_idxs_[bin_idx];
  }

  const T radius_;
  const T bin_dim_;
  const int32_t bin_radius_;
  const int32_t grid_stride_;
  const int32_t num_grid_bins_;
  int32_t num_valid_grid_bins_;
  int32_t num_edge_bins_;
  std::vector<GridCell<T>> grid_cells_;
  std::vector<int32_t> distance_bin_permuted_idxs_;
};

template <typename T>
struct GridCellEnriched {
  T z_elevation;
  T max_obscured_slope;
  T ray_distance_squared;
};

struct PartitionInfo {
  static constexpr const int32_t invalid_partition_idx{-1};
  const int32_t num_primary_partitions;
  const int32_t num_secondary_partitions;
  const int32_t num_partitions;
  const int32_t num_elems;
  std::vector<std::pair<int32_t, int32_t>>
      partition_offsets_and_sizes;  // dense grid, has entries even for empty partitions
                                    // (denoted by 0 size)
  std::vector<int32_t> permuted_idxs;
  std::vector<int32_t> idx_partition_map;

  int32_t num_non_sparse_partitions{0};
  int32_t first_valid_element{0};

  PartitionInfo(const size_t num_primary_partitions,
                const size_t num_secondary_partitions,
                const size_t num_elems)
      : num_primary_partitions(num_primary_partitions)
      , num_secondary_partitions(num_secondary_partitions)
      , num_partitions(num_primary_partitions * num_secondary_partitions)
      , num_elems(num_elems)
      , partition_offsets_and_sizes(num_partitions, std::make_pair(-1, 0))
      , permuted_idxs(num_elems)
      , idx_partition_map(num_elems) {}

  inline int32_t get_permuted_idx(const int32_t input_idx) const {
    return permuted_idxs[input_idx];
  }

  inline int32_t get_partition_for_unpermuted_idx(const int32_t unpermuted_idx) const {
    return idx_partition_map[get_permuted_idx(unpermuted_idx)];
  }

  inline int32_t get_partition_for_permuted_idx(const int32_t permuted_idx) const {
    return idx_partition_map[permuted_idx];
  }

  void permute_by_partition();

  void fill_partition_offsets_and_sizes();
};

template <typename T>
RasterOffsets<T>::RasterOffsets(const double radius,
                                const double bin_dim,
                                const bool only_first_quarter_radian)
    : radius_(radius)
    , bin_dim_(bin_dim)
    , bin_radius_(radius / bin_dim)
    , grid_stride_(bin_radius_ * 2 + 1)
    , num_grid_bins_(grid_stride_ * grid_stride_)
    , distance_bin_permuted_idxs_(num_grid_bins_, 0) {
  auto timer = DEBUG_TIMER(__func__);
  std::vector<GridCell<T>> temp_grid_cells;

  const int32_t min_x_bin_idx = only_first_quarter_radian ? 0 : -bin_radius_;
  for (int32_t x_bin_idx = min_x_bin_idx; x_bin_idx <= bin_radius_; ++x_bin_idx) {
    const int32_t min_y_bin_idx = only_first_quarter_radian ? 0 : -bin_radius_;
    const int32_t max_y_bin_idx = only_first_quarter_radian ? x_bin_idx : bin_radius_;
    for (int32_t y_bin_idx = min_y_bin_idx; y_bin_idx <= max_y_bin_idx; ++y_bin_idx) {
      const int32_t bin_idx = x_y_bin_to_bin_index(
          x_bin_idx + bin_radius_, y_bin_idx + bin_radius_, grid_stride_);

      const T unit_distance = sqrt(x_bin_idx * x_bin_idx + y_bin_idx * y_bin_idx);
      const T xy_distance = std::max(unit_distance * bin_dim_, static_cast<T>(0.01));
      const T inverted_xy_distance = 1.0 / xy_distance;
      T xy_angle = 0;
      if (xy_distance <= radius_) {
        xy_angle = xy_distance > 0. ? atan2(y_bin_idx, x_bin_idx) : 0.;
        int32_t parent_idx = 0;
        if (xy_distance > 0) {
          const T x_ratio = static_cast<T>(x_bin_idx) / unit_distance;
          const T y_ratio = static_cast<T>(y_bin_idx) / unit_distance;
          const int32_t parent_x_bin_idx = x_bin_idx - x_ratio;
          const int32_t parent_y_bin_idx = y_bin_idx - y_ratio;
          parent_idx = x_y_bin_to_bin_index(parent_x_bin_idx + bin_radius_,
                                            parent_y_bin_idx + bin_radius_,
                                            grid_stride_);
        }

        const GridCell<T> grid_cell{bin_idx,
                                    x_bin_idx,
                                    y_bin_idx,
                                    xy_distance,
                                    inverted_xy_distance,
                                    xy_angle,
                                    parent_idx};

        temp_grid_cells.emplace_back(grid_cell);
      }
    }
  }

  std::sort(temp_grid_cells.begin(), temp_grid_cells.end(), sort_by_distance<T>);
  std::swap(grid_cells_, temp_grid_cells);  // shrink to fit
  num_valid_grid_bins_ = grid_cells_.size();
  for (int32_t i = 0; i != num_valid_grid_bins_; ++i) {
    distance_bin_permuted_idxs_[grid_cells_[i].bin_idx] = i;
    if (grid_cells_[i].parent_idx >= 0) {
      grid_cells_[i].parent_idx = distance_bin_permuted_idxs_[grid_cells_[i].parent_idx];
    }
  }
  const T edge_distance_with_epsilon = radius_ - bin_dim_ + (bin_dim_ * 0.1);
  num_edge_bins_ = 0;
  for (int32_t i = num_valid_grid_bins_ - 1; i >= 0; --i) {
    if (grid_cells_[i].xy_distance > edge_distance_with_epsilon) {
      num_edge_bins_++;
    }
  }
}

template <typename T>
PartitionInfo get_partition_info(const Column<T>& rf_source_sort_primary_dim,
                                 const Column<T>& rf_source_sort_secondary_dim,
                                 const T primary_dim_min,
                                 const T primary_dim_scale,
                                 const T secondary_dim_min,
                                 const T secondary_dim_scale,
                                 const int32_t partition_size,
                                 const int32_t num_primary_partitions,
                                 const int32_t num_secondary_partitions) {
  const int64_t num_elems{rf_source_sort_primary_dim.size()};
  PartitionInfo partition_info(
      num_primary_partitions, num_secondary_partitions, num_elems);
  for (int64_t idx = 0; idx < num_elems; ++idx) {
    partition_info.permuted_idxs[idx] = idx;
    const int32_t primary_source_bin =
        (rf_source_sort_primary_dim[idx] - primary_dim_min) * primary_dim_scale;
    const int32_t source_x_partition = primary_source_bin / partition_size;
    const int32_t secondary_source_bin =
        (rf_source_sort_secondary_dim[idx] - secondary_dim_min) * secondary_dim_scale;
    const int32_t source_y_partition = secondary_source_bin / partition_size;
    if (source_x_partition < 0 || source_x_partition >= num_primary_partitions ||
        source_y_partition < 0 || source_y_partition >= num_secondary_partitions) {
      partition_info.idx_partition_map[idx] = PartitionInfo::invalid_partition_idx;
      continue;
    }
    partition_info.idx_partition_map[idx] = x_y_bin_to_bin_index(
        source_x_partition, source_y_partition, num_primary_partitions);
  }

  partition_info.permute_by_partition();
  partition_info.fill_partition_offsets_and_sizes();
  return partition_info;
}

template <typename S, typename T>
void write_local_output(Column<T>& out_min_squared_distances,
                        Column<S>& out_max_signal_strength_source_ids,
                        std::vector<T>& local_min_squared_distances,
                        std::vector<S>& local_max_signal_strength_source_ids,
                        const std::pair<int64_t, int64_t>& centroid_partition,
                        const int64_t partition_size,
                        const int64_t num_x_partitions,
                        const int64_t num_y_partitions,
                        const int64_t num_global_x_bins,
                        const int64_t num_global_y_bins,
                        const T numeric_null_sentinel,
                        std::vector<std::mutex>& spatial_output_mutexes) {
  const int64_t num_local_x_bins{partition_size * 3};

  const int64_t global_x_bin_offset = (centroid_partition.first - 1) * partition_size;
  const int64_t global_y_bin_offset = (centroid_partition.second - 1) * partition_size;
  for (int64_t x_partition_idx = centroid_partition.first - 1;
       x_partition_idx <= centroid_partition.first + 1;
       ++x_partition_idx) {
    for (int64_t y_partition_idx = centroid_partition.second - 1;
         y_partition_idx <= centroid_partition.second + 1;
         ++y_partition_idx) {
      if (x_partition_idx < 0 || x_partition_idx >= num_x_partitions ||
          y_partition_idx < 0 || y_partition_idx >= num_y_partitions) {
        continue;
      }
      const int64_t partition_idx =
          x_y_bin_to_bin_index(x_partition_idx, y_partition_idx, num_x_partitions);
      const int64_t start_x_bin_idx =
          (x_partition_idx - centroid_partition.first + 1) * partition_size;
      const int64_t start_y_bin_idx =
          (y_partition_idx - centroid_partition.second + 1) * partition_size;
      const std::lock_guard<std::mutex> partition_write_lock(
          spatial_output_mutexes[partition_idx]);
      for (int64_t y_bin_idx = start_y_bin_idx;
           y_bin_idx < start_y_bin_idx + partition_size;
           ++y_bin_idx) {
        const int64_t global_y_bin_idx = y_bin_idx + global_y_bin_offset;
        if (global_y_bin_idx >= num_global_y_bins) {
          continue;
        }
        for (int64_t x_bin_idx = start_x_bin_idx;
             x_bin_idx < start_x_bin_idx + partition_size;
             ++x_bin_idx) {
          const int64_t global_x_bin_idx = x_bin_idx + global_x_bin_offset;
          if (global_x_bin_idx >= num_global_x_bins) {
            continue;
          }
          const int64_t global_bin_idx =
              x_y_bin_to_bin_index(global_x_bin_idx, global_y_bin_idx, num_global_x_bins);
          const int64_t local_bin_idx =
              x_y_bin_to_bin_index(x_bin_idx, y_bin_idx, num_local_x_bins);

          const T local_min_squared_distance = local_min_squared_distances[local_bin_idx];
          if (local_min_squared_distance == numeric_null_sentinel) {
            continue;
          }
          if (local_min_squared_distance < out_min_squared_distances[global_bin_idx]) {
            out_min_squared_distances[global_bin_idx] = local_min_squared_distance;
            out_max_signal_strength_source_ids[global_bin_idx] =
                local_max_signal_strength_source_ids[local_bin_idx];
          }
        }
      }
    }
  }
}

template <typename S, typename T>
void write_local_signal_dbm_output(Column<T>& out_max_signal_strength_dbm,
                                   Column<S>& out_max_signal_strength_source_ids,
                                   std::vector<T>& local_max_signal_dbms,
                                   std::vector<S>& local_max_signal_dbms_source_ids,
                                   const std::pair<int64_t, int64_t>& centroid_partition,
                                   const int64_t partition_size,
                                   const int64_t num_x_partitions,
                                   const int64_t num_y_partitions,
                                   const int64_t num_global_x_bins,
                                   const int64_t num_global_y_bins,
                                   const T numeric_null_sentinel,
                                   std::vector<std::mutex>& spatial_output_mutexes) {
  const int64_t num_local_x_bins{partition_size * 3};

  const int64_t global_x_bin_offset = (centroid_partition.first - 1) * partition_size;
  const int64_t global_y_bin_offset = (centroid_partition.second - 1) * partition_size;
  for (int64_t x_partition_idx = centroid_partition.first - 1;
       x_partition_idx <= centroid_partition.first + 1;
       ++x_partition_idx) {
    for (int64_t y_partition_idx = centroid_partition.second - 1;
         y_partition_idx <= centroid_partition.second + 1;
         ++y_partition_idx) {
      if (x_partition_idx < 0 || x_partition_idx >= num_x_partitions ||
          y_partition_idx < 0 || y_partition_idx >= num_y_partitions) {
        continue;
      }
      const int64_t partition_idx =
          x_y_bin_to_bin_index(x_partition_idx, y_partition_idx, num_x_partitions);
      const int64_t start_x_bin_idx =
          (x_partition_idx - centroid_partition.first + 1) * partition_size;
      const int64_t start_y_bin_idx =
          (y_partition_idx - centroid_partition.second + 1) * partition_size;
      const std::lock_guard<std::mutex> partition_write_lock(
          spatial_output_mutexes[partition_idx]);
      for (int64_t y_bin_idx = start_y_bin_idx;
           y_bin_idx < start_y_bin_idx + partition_size;
           ++y_bin_idx) {
        const int64_t global_y_bin_idx = y_bin_idx + global_y_bin_offset;
        if (global_y_bin_idx >= num_global_y_bins) {
          continue;
        }
        for (int64_t x_bin_idx = start_x_bin_idx;
             x_bin_idx < start_x_bin_idx + partition_size;
             ++x_bin_idx) {
          const int64_t global_x_bin_idx = x_bin_idx + global_x_bin_offset;
          if (global_x_bin_idx >= num_global_x_bins) {
            continue;
          }
          const int64_t global_bin_idx =
              x_y_bin_to_bin_index(global_x_bin_idx, global_y_bin_idx, num_global_x_bins);
          const int64_t local_bin_idx =
              x_y_bin_to_bin_index(x_bin_idx, y_bin_idx, num_local_x_bins);

          const T local_max_signal_dbm = local_max_signal_dbms[local_bin_idx];
          if (local_max_signal_dbm == numeric_null_sentinel) {
            continue;
          }
          if (local_max_signal_dbm > out_max_signal_strength_dbm[global_bin_idx]) {
            out_max_signal_strength_dbm[global_bin_idx] = local_max_signal_dbm;
            out_max_signal_strength_source_ids[global_bin_idx] =
                local_max_signal_dbms_source_ids[local_bin_idx];
          }
        }
      }
    }
  }
}

template <typename T1>
T1 get_max_ray_step(const int32_t num_ray_steps,
                    const T1 angle_radians_step,
                    const T1 rays_per_bin_autosplit_threshold,
                    const T1 ray_step_in_bins) {
  if (rays_per_bin_autosplit_threshold <= 0.0) {
    return num_ray_steps;
  }
  const T1 num_rays = 2 * M_PI / angle_radians_step;
  const int32_t target_split_bin =
      num_rays / (M_PI * 2.0) / rays_per_bin_autosplit_threshold;
  const int32_t target_split_step = target_split_bin / ray_step_in_bins;
  return std::min(num_ray_steps, target_split_step);
}

double get_min_distance_for_power_frequency(const double receiver_power_threshold_dbm,
                                            const double source_power_dbm,
                                            const double signal_frequency_mhz);

double get_min_distance_for_power_frequency_antenna_max_gain(
    const double receiver_power_threshold_dbm,
    const double source_power_dbm,
    const double signal_frequency_mhz,
    const double antenna_max_gain_dbm);

template <typename T>
T get_horizontal_tx_power_component(const AntennaPattern& antenna_pattern,
                                    const T source_antenna_azimuth_degrees,
                                    const T angle_radians) {
  if (antenna_pattern.is_isotropic) {
    return 0.0;
  }
  T bearing_angle_degrees = std::fmod(90.0 - (angle_radians * 180.0 / M_PI), 360.0);
  if (bearing_angle_degrees < 0) {
    bearing_angle_degrees = 360.0 + bearing_angle_degrees;
  }
  return antenna_pattern.get_directional_tx_power_component(
      source_antenna_azimuth_degrees, bearing_angle_degrees, AntennaAxis::HORIZONTAL);
}

template <typename S, typename T1, typename Z1>
void propagate_ray(const S source_id,
                   const T1 start_x_bin,
                   const T1 start_y_bin,
                   const Z1 source_z,
                   const T1 source_power_dbm,
                   const T1 source_freq_mhz,
                   const T1 source_antenna_azimuth_degrees,
                   const T1 source_antenna_downtilt_degrees,
                   const AntennaPattern& antenna_pattern,
                   const int32_t global_x_bin_offset,
                   const int32_t global_y_bin_offset,
                   const double assumed_receiver_height_agl,
                   const double assumed_ground_z,
                   const T1 angle_radians,
                   const T1 angle_radians_step,
                   const T1 rays_per_bin_autosplit_threshold,
                   const int32_t start_ray_step,
                   const int32_t num_ray_steps,
                   const T1 starting_max_obscured_z_slope,
                   const T1 ray_step_in_bins,
                   const int32_t num_local_bins_per_dim,
                   const T1 min_receiver_signal_strength_dbm,
                   const Column<Z1>& out_max_z,
                   const GeoRaster<T1, Z1>& geo_raster,
                   std::vector<T1>& local_max_signal_dbms,
                   std::vector<S>& local_max_signal_dbms_source_ids) {
  const auto horizontal_direction_tx_power_component = get_horizontal_tx_power_component(
      antenna_pattern, source_antenna_azimuth_degrees, angle_radians);

  const T1 log_10_freq = log10(source_freq_mhz);
  const int32_t max_ray_step = get_max_ray_step(num_ray_steps,
                                                angle_radians_step,
                                                rays_per_bin_autosplit_threshold,
                                                ray_step_in_bins);
  const T1 ray_step_meters = ray_step_in_bins * geo_raster.bin_dim_meters_;
  const int64_t num_x_bins = geo_raster.num_x_bins_;
  const T1 ray_x_step = cos(angle_radians) * ray_step_in_bins;
  const T1 ray_y_step = sin(angle_radians) * ray_step_in_bins;
  T1 ray_step_z = assumed_ground_z;
  T1 max_obscured_z_slope = starting_max_obscured_z_slope;
  for (int32_t s = start_ray_step; s < max_ray_step; ++s) {
    const T1 ray_step_with_epsilon = s + ray_step_epsilon;
    const int32_t ray_step_x_bin =
        std::floor(start_x_bin + ray_step_with_epsilon * ray_x_step);
    const int32_t ray_step_y_bin =
        std::floor(start_y_bin + ray_step_with_epsilon * ray_y_step);
    if (geo_raster.is_bin_out_of_bounds(ray_step_x_bin, ray_step_y_bin)) {
      // Off grid, won't ever get back in so move to next ray
      break;
    }
    const int32_t ray_step_raster_idx =
        x_y_bin_to_bin_index(ray_step_x_bin, ray_step_y_bin, num_x_bins);

    if (!out_max_z.isNull(ray_step_raster_idx)) {
      ray_step_z = out_max_z[ray_step_raster_idx];
    } else {
      break;
    }
    const T1 xy_meters_from_source = ray_step_with_epsilon * ray_step_meters;
    const T1 receiver_z_meters_from_source =
        ray_step_z + assumed_receiver_height_agl - source_z;
    const T1 ray_step_receiver_z_slope =
        receiver_z_meters_from_source / xy_meters_from_source;
    if (ray_step_receiver_z_slope < max_obscured_z_slope) {
      continue;
    }
    const T1 terrain_z_meters_from_source = ray_step_z - source_z;
    const T1 ray_step_terrain_z_slope =
        terrain_z_meters_from_source / xy_meters_from_source;
    if (ray_step_terrain_z_slope > max_obscured_z_slope) {
      max_obscured_z_slope = ray_step_terrain_z_slope;
    }

    const auto vertical_direction_tx_power_component =
        antenna_pattern.is_isotropic
            ? 0.0
            : antenna_pattern.get_directional_tx_power_component(
                  -source_antenna_downtilt_degrees,
                  static_cast<T1>(std::atan(ray_step_receiver_z_slope) * 180.0 / M_PI),
                  AntennaAxis::VERTICAL);
    const T1 ray_step_distance_squared =
        xy_meters_from_source * xy_meters_from_source +
        receiver_z_meters_from_source * receiver_z_meters_from_source;
    const int32_t local_x_bin_idx = ray_step_x_bin - global_x_bin_offset;
    const int32_t local_y_bin_idx = ray_step_y_bin - global_y_bin_offset;
    const int32_t local_bin_idx =
        x_y_bin_to_bin_index(local_x_bin_idx, local_y_bin_idx, num_local_bins_per_dim);
    const auto local_rf_signal_dbm =
        source_power_dbm + antenna_pattern.gain +
        horizontal_direction_tx_power_component + vertical_direction_tx_power_component -
        (20.0 * log10(sqrt(ray_step_distance_squared)) + 20.0 * log_10_freq - 27.55);
    if (local_rf_signal_dbm < min_receiver_signal_strength_dbm) {
      return;
    }
    if (local_rf_signal_dbm > local_max_signal_dbms[local_bin_idx]) {
      local_max_signal_dbms[local_bin_idx] = local_rf_signal_dbm;
      local_max_signal_dbms_source_ids[local_bin_idx] = source_id;
    }
  }
  if (max_ray_step < num_ray_steps) {
    const T1 new_angle_step = angle_radians_step * 0.5;
    propagate_ray(source_id,
                  start_x_bin,
                  start_y_bin,
                  source_z,
                  source_power_dbm,
                  source_freq_mhz,
                  source_antenna_azimuth_degrees,
                  source_antenna_downtilt_degrees,
                  antenna_pattern,
                  global_x_bin_offset,
                  global_y_bin_offset,
                  assumed_receiver_height_agl,
                  ray_step_z,
                  angle_radians,
                  new_angle_step,
                  rays_per_bin_autosplit_threshold,
                  max_ray_step,
                  num_ray_steps,
                  max_obscured_z_slope,
                  ray_step_in_bins,
                  num_local_bins_per_dim,
                  min_receiver_signal_strength_dbm,
                  out_max_z,
                  geo_raster,
                  local_max_signal_dbms,
                  local_max_signal_dbms_source_ids);
    propagate_ray(source_id,
                  start_x_bin,
                  start_y_bin,
                  source_z,
                  source_power_dbm,
                  source_freq_mhz,
                  source_antenna_azimuth_degrees,
                  source_antenna_downtilt_degrees,
                  antenna_pattern,
                  global_x_bin_offset,
                  global_y_bin_offset,
                  assumed_receiver_height_agl,
                  ray_step_z,
                  angle_radians + new_angle_step,
                  new_angle_step,
                  rays_per_bin_autosplit_threshold,
                  max_ray_step,
                  num_ray_steps,
                  max_obscured_z_slope,
                  ray_step_in_bins,
                  num_local_bins_per_dim,
                  min_receiver_signal_strength_dbm,
                  out_max_z,
                  geo_raster,
                  local_max_signal_dbms,
                  local_max_signal_dbms_source_ids);
  }
}

template <typename S, typename T1, typename Z1>
void propagate_ray_with_attenuation(
    const S source_id,
    const T1 start_x_bin,
    const T1 start_y_bin,
    const Z1 source_z,
    const T1 source_power_dbm,
    const T1 source_freq_mhz,
    const T1 source_antenna_azimuth_degrees,
    const T1 source_antenna_downtilt_degrees,
    const AntennaPattern& antenna_pattern,
    const int32_t global_x_bin_offset,
    const int32_t global_y_bin_offset,
    const double assumed_receiver_height_agl,
    const double assumed_ground_z,
    const double assumed_terrain_z,
    const T1 angle_radians,
    const T1 angle_radians_step,
    const T1 rays_per_bin_autosplit_threshold,
    const int32_t start_ray_step,
    const int32_t num_ray_steps,
    const T1 starting_max_obscured_z_slope,
    const T1 ray_step_in_bins,
    const int32_t num_local_bins_per_dim,
    const T1 min_receiver_signal_strength_dbm,
    const Column<Z1>& ground_z,
    const Column<Z1>& terrain_z,
    const Column<Z1>& terrain_attenuation_dbm_per_meter,
    const GeoRaster<T1, Z1>& geo_raster,
    AngleAttenuationsFlatMap<T1> vertical_angle_attenuations_dbm,
    std::vector<T1>& local_max_signal_dbms,
    std::vector<S>& local_max_signal_dbms_source_ids) {
  const auto horizontal_direction_tx_power_component = get_horizontal_tx_power_component(
      antenna_pattern, source_antenna_azimuth_degrees, angle_radians);

  const T1 log_10_freq = log10(source_freq_mhz);
  const int32_t max_ray_step = get_max_ray_step(num_ray_steps,
                                                angle_radians_step,
                                                rays_per_bin_autosplit_threshold,
                                                ray_step_in_bins);
  const T1 ray_step_meters = ray_step_in_bins * geo_raster.bin_dim_meters_;
  const int64_t num_x_bins = geo_raster.num_x_bins_;
  const T1 ray_x_step = cos(angle_radians) * ray_step_in_bins;
  const T1 ray_y_step = sin(angle_radians) * ray_step_in_bins;
  T1 ray_step_terrain_z = assumed_terrain_z;
  T1 ray_step_ground_z = assumed_ground_z;
  T1 ray_step_attenuation_dbm_per_meter = 0.0;
  T1 max_obscured_z_slope = starting_max_obscured_z_slope;
  for (int32_t s = start_ray_step; s < max_ray_step; ++s) {
    const T1 ray_step_with_epsilon = s + ray_step_epsilon;
    const int32_t ray_step_x_bin =
        std::floor(start_x_bin + ray_step_with_epsilon * ray_x_step);
    const int32_t ray_step_y_bin =
        std::floor(start_y_bin + ray_step_with_epsilon * ray_y_step);
    if (geo_raster.is_bin_out_of_bounds(ray_step_x_bin, ray_step_y_bin)) {
      // Off grid, won't ever get back in so move to next ray
      break;
    }
    const int32_t ray_step_raster_idx =
        x_y_bin_to_bin_index(ray_step_x_bin, ray_step_y_bin, num_x_bins);

    if (!ground_z.isNull(ray_step_raster_idx)) {
      ray_step_ground_z = ground_z[ray_step_raster_idx];
    }
    if (!terrain_z.isNull(ray_step_raster_idx)) {
      ray_step_terrain_z = terrain_z[ray_step_raster_idx];
    }
    if (!terrain_attenuation_dbm_per_meter.isNull(ray_step_raster_idx)) {
      ray_step_attenuation_dbm_per_meter =
          terrain_attenuation_dbm_per_meter[ray_step_raster_idx];
    }

    const T1 xy_meters_from_source = ray_step_with_epsilon * ray_step_meters;
    const T1 receiver_z_meters_from_source =
        ray_step_ground_z + assumed_receiver_height_agl - source_z;
    const T1 ray_step_receiver_z_slope =
        receiver_z_meters_from_source / xy_meters_from_source;

    const T1 terrain_z_meters_from_source = ray_step_terrain_z - source_z;
    const T1 ray_step_terrain_z_slope =
        terrain_z_meters_from_source / xy_meters_from_source;

    const T1 ground_z_meters_from_source = ray_step_ground_z - source_z;
    const T1 ray_step_ground_z_slope =
        ground_z_meters_from_source / xy_meters_from_source;

    const T1 new_max_obscured_slope =
        std::max(max_obscured_z_slope, ray_step_ground_z_slope);
    if (ray_step_terrain_z_slope > new_max_obscured_slope) {
      const T1 obscured_theta = atan(new_max_obscured_slope);
      const T1 terrain_theta = atan(ray_step_terrain_z_slope);
      const T1 attenuation_dbm =
          ray_step_attenuation_dbm_per_meter * geo_raster.bin_dim_meters_;
      vertical_angle_attenuations_dbm.add_attenuations(
          obscured_theta, terrain_theta, attenuation_dbm);
    }

    if (ray_step_receiver_z_slope < max_obscured_z_slope) {
      continue;
    }
    if (ray_step_ground_z_slope > max_obscured_z_slope) {
      max_obscured_z_slope = ray_step_ground_z_slope;
    }

    const auto vertical_direction_tx_power_component =
        antenna_pattern.is_isotropic
            ? 0.0
            : antenna_pattern.get_directional_tx_power_component(
                  -source_antenna_downtilt_degrees,
                  static_cast<T1>(atan(ray_step_receiver_z_slope) * 180.0 / M_PI),
                  AntennaAxis::VERTICAL);
    const T1 ray_step_distance =
        sqrt(xy_meters_from_source * xy_meters_from_source +
             receiver_z_meters_from_source * receiver_z_meters_from_source);
    const int32_t local_x_bin_idx = ray_step_x_bin - global_x_bin_offset;
    const int32_t local_y_bin_idx = ray_step_y_bin - global_y_bin_offset;
    const int32_t local_bin_idx =
        x_y_bin_to_bin_index(local_x_bin_idx, local_y_bin_idx, num_local_bins_per_dim);
    const auto local_rf_signal_dbm_without_attenuation =
        source_power_dbm + antenna_pattern.gain +
        horizontal_direction_tx_power_component + vertical_direction_tx_power_component -
        (20.0 * log10(ray_step_distance) + 20.0 * log_10_freq - 27.55);
    if (local_rf_signal_dbm_without_attenuation < min_receiver_signal_strength_dbm) {
      return;
    }
    const T1 receiver_theta = atan(ray_step_receiver_z_slope);
    // Scale receiver attenuation according to actual travel distance of the ray
    const T1 receiver_attenuation_dbm =
        vertical_angle_attenuations_dbm.get_angle_attenuation(receiver_theta) *
        (ray_step_distance / xy_meters_from_source);
    const auto local_rf_signal_dbm =
        local_rf_signal_dbm_without_attenuation - receiver_attenuation_dbm;

    if (local_rf_signal_dbm >= min_receiver_signal_strength_dbm &&
        local_rf_signal_dbm > local_max_signal_dbms[local_bin_idx]) {
      local_max_signal_dbms[local_bin_idx] = local_rf_signal_dbm;
      local_max_signal_dbms_source_ids[local_bin_idx] = source_id;
    }
  }
  if (max_ray_step < num_ray_steps) {
    const T1 new_angle_step = angle_radians_step * 0.5;
    propagate_ray_with_attenuation(source_id,
                                   start_x_bin,
                                   start_y_bin,
                                   source_z,
                                   source_power_dbm,
                                   source_freq_mhz,
                                   source_antenna_azimuth_degrees,
                                   source_antenna_downtilt_degrees,
                                   antenna_pattern,
                                   global_x_bin_offset,
                                   global_y_bin_offset,
                                   assumed_receiver_height_agl,
                                   ray_step_ground_z,
                                   ray_step_terrain_z,
                                   angle_radians,
                                   new_angle_step,
                                   rays_per_bin_autosplit_threshold,
                                   max_ray_step,
                                   num_ray_steps,
                                   max_obscured_z_slope,
                                   ray_step_in_bins,
                                   num_local_bins_per_dim,
                                   min_receiver_signal_strength_dbm,
                                   ground_z,
                                   terrain_z,
                                   terrain_attenuation_dbm_per_meter,
                                   geo_raster,
                                   vertical_angle_attenuations_dbm,
                                   local_max_signal_dbms,
                                   local_max_signal_dbms_source_ids);
    propagate_ray_with_attenuation(source_id,
                                   start_x_bin,
                                   start_y_bin,
                                   source_z,
                                   source_power_dbm,
                                   source_freq_mhz,
                                   source_antenna_azimuth_degrees,
                                   source_antenna_downtilt_degrees,
                                   antenna_pattern,
                                   global_x_bin_offset,
                                   global_y_bin_offset,
                                   assumed_receiver_height_agl,
                                   ray_step_ground_z,
                                   ray_step_terrain_z,
                                   angle_radians + new_angle_step,
                                   new_angle_step,
                                   rays_per_bin_autosplit_threshold,
                                   max_ray_step,
                                   num_ray_steps,
                                   max_obscured_z_slope,
                                   ray_step_in_bins,
                                   num_local_bins_per_dim,
                                   min_receiver_signal_strength_dbm,
                                   ground_z,
                                   terrain_z,
                                   terrain_attenuation_dbm_per_meter,
                                   geo_raster,
                                   vertical_angle_attenuations_dbm,
                                   local_max_signal_dbms,
                                   local_max_signal_dbms_source_ids);
  }
}

template <typename S, typename T1, typename Z1>
struct FlattenedAntennaParams {
  FlattenedAntennaParams(
      TableFunctionManager& mgr,
      const Column<Array<S>>& rf_source_id,
      const Column<Array<TextEncodingDict>>& rf_source_antenna_type,
      std::vector<S>& rf_source_id_exploded_vec,
      std::vector<T1>& rf_source_x_exploded_vec,
      std::vector<T1>& rf_source_y_exploded_vec,
      std::vector<Z1>& rf_source_z_exploded_vec,
      std::vector<T1>& rf_source_power_watts_exploded_vec,
      std::vector<T1>& rf_source_freq_mhz_exploded_vec,
      std::vector<T1>& rf_source_antenna_azimuth_degrees_exploded_vec,
      std::vector<T1>& rf_source_antenna_downtilt_degrees_exploded_vec,
      std::vector<TextEncodingDict>& rf_source_antenna_type_exploded_vec)
      : rf_source_id_exploded(rf_source_id_exploded_vec)
      , rf_source_x_exploded(rf_source_x_exploded_vec)
      , rf_source_y_exploded(rf_source_y_exploded_vec)
      , rf_source_z_exploded(rf_source_z_exploded_vec)
      , rf_source_power_watts_exploded(rf_source_power_watts_exploded_vec)
      , rf_source_freq_mhz_exploded(rf_source_freq_mhz_exploded_vec)
      , rf_source_antenna_azimuth_degrees_exploded(
            rf_source_antenna_azimuth_degrees_exploded_vec)
      , rf_source_antenna_downtilt_degrees_exploded(
            rf_source_antenna_downtilt_degrees_exploded_vec)
      , rf_source_antenna_type_exploded(rf_source_antenna_type_exploded_vec) {
    if constexpr (std::is_same<S, TextEncodingDict>::value) {
      // If S is of type TextEncodingDict, we need to copy over the string
      // dictionary proxy pointer to the new synthesized column
      const auto dict_db_id = rf_source_id.getDictDbId();
      const auto dict_id = rf_source_id.getDictId();
      rf_source_id_exploded.string_dict_proxy_ =
          mgr.getStringDictionaryProxy(dict_db_id, dict_id);
    }
    const auto antenna_type_dict_db_id = rf_source_antenna_type.getDictDbId();
    const auto antenna_type_dict_id = rf_source_antenna_type.getDictId();
    rf_source_antenna_type_exploded.string_dict_proxy_ =
        mgr.getStringDictionaryProxy(antenna_type_dict_db_id, antenna_type_dict_id);
  }

  Column<S> rf_source_id_exploded;
  Column<T1> rf_source_x_exploded;
  Column<T1> rf_source_y_exploded;
  Column<Z1> rf_source_z_exploded;
  Column<T1> rf_source_power_watts_exploded;
  Column<T1> rf_source_freq_mhz_exploded;
  Column<T1> rf_source_antenna_azimuth_degrees_exploded;
  Column<T1> rf_source_antenna_downtilt_degrees_exploded;
  Column<TextEncodingDict> rf_source_antenna_type_exploded;
};

template <typename S, typename T1, typename Z1>
class AntennaArrayFlattener {
 public:
  AntennaArrayFlattener(const Column<Array<S>>& rf_source_id,
                        const Column<T1>& rf_source_x,
                        const Column<T1>& rf_source_y,
                        const Column<Z1>& rf_source_z,
                        const Column<Array<T1>>& rf_source_power_watts,
                        const Column<Array<T1>>& rf_source_freq_mhz,
                        const Column<Array<T1>>& rf_source_antenna_azimuth_degrees,
                        const Column<Array<T1>>& rf_source_antenna_downtilt_degrees,
                        const Column<Array<TextEncodingDict>>& rf_source_antenna_type) {
    const int64_t num_towers = rf_source_id.size();
    CHECK_EQ(num_towers, rf_source_antenna_azimuth_degrees.size());

    std::vector<int64_t> tower_antennas_prefix_sum(num_towers + 1);
    tower_antennas_prefix_sum[0] = 0;

    for (int64_t tower_idx = 0; tower_idx < num_towers; ++tower_idx) {
      const int64_t tower_num_antennas = rf_source_id[tower_idx].getSize();

      if (tower_num_antennas != rf_source_power_watts[tower_idx].getSize() ||
          tower_num_antennas != rf_source_freq_mhz[tower_idx].getSize() ||
          tower_num_antennas != rf_source_antenna_azimuth_degrees[tower_idx].getSize() ||
          tower_num_antennas != rf_source_antenna_downtilt_degrees[tower_idx].getSize() ||
          tower_num_antennas != rf_source_antenna_type[tower_idx].getSize()) {
        throw std::runtime_error("Tower antenna arrays not equal in length");
      }
      tower_antennas_prefix_sum[tower_idx + 1] =
          tower_antennas_prefix_sum[tower_idx] + tower_num_antennas;
    }
    const int64_t num_antennas = tower_antennas_prefix_sum[num_towers];
    rf_source_id_exploded_vec_.resize(num_antennas);
    rf_source_x_exploded_vec_.resize(num_antennas);
    rf_source_y_exploded_vec_.resize(num_antennas);
    rf_source_z_exploded_vec_.resize(num_antennas);
    rf_source_power_watts_exploded_vec_.resize(num_antennas);
    rf_source_freq_mhz_exploded_vec_.resize(num_antennas);
    rf_source_antenna_azimuth_degrees_exploded_vec_.resize(num_antennas);
    rf_source_antenna_downtilt_degrees_exploded_vec_.resize(num_antennas);
    rf_source_antenna_type_exploded_vec_.resize(num_antennas);
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, num_towers),
        [&](const tbb::blocked_range<int64_t>& r) {
          const auto start_tower_idx = r.begin();
          const auto end_tower_idx = r.end();
          for (int64_t tower_idx = start_tower_idx; tower_idx < end_tower_idx;
               ++tower_idx) {
            const int64_t start_antenna_idx = tower_antennas_prefix_sum[tower_idx];
            const int64_t end_antenna_idx = tower_antennas_prefix_sum[tower_idx + 1];
            for (int64_t antenna_idx = start_antenna_idx; antenna_idx < end_antenna_idx;
                 ++antenna_idx) {
              rf_source_id_exploded_vec_[antenna_idx] =
                  rf_source_id[tower_idx][antenna_idx - start_antenna_idx];
              rf_source_x_exploded_vec_[antenna_idx] = rf_source_x[tower_idx];
              rf_source_y_exploded_vec_[antenna_idx] = rf_source_y[tower_idx];
              rf_source_z_exploded_vec_[antenna_idx] = rf_source_z[tower_idx];
              rf_source_power_watts_exploded_vec_[antenna_idx] =
                  rf_source_power_watts[tower_idx][antenna_idx - start_antenna_idx];
              rf_source_freq_mhz_exploded_vec_[antenna_idx] =
                  rf_source_freq_mhz[tower_idx][antenna_idx - start_antenna_idx];
              rf_source_antenna_azimuth_degrees_exploded_vec_[antenna_idx] =
                  rf_source_antenna_azimuth_degrees[tower_idx]
                                                   [antenna_idx - start_antenna_idx];
              rf_source_antenna_downtilt_degrees_exploded_vec_[antenna_idx] =
                  rf_source_antenna_downtilt_degrees[tower_idx]
                                                    [antenna_idx - start_antenna_idx];
              rf_source_antenna_type_exploded_vec_[antenna_idx] =
                  rf_source_antenna_type[tower_idx][antenna_idx - start_antenna_idx];
            }
          }
        });
  }

  FlattenedAntennaParams<S, T1, Z1> getFlattenedAntennaParams(
      TableFunctionManager& mgr,
      const Column<Array<S>>& rf_source_id,
      const Column<Array<TextEncodingDict>>& rf_source_antenna_type) {
    return FlattenedAntennaParams(mgr,
                                  rf_source_id,
                                  rf_source_antenna_type,
                                  rf_source_id_exploded_vec_,
                                  rf_source_x_exploded_vec_,
                                  rf_source_y_exploded_vec_,
                                  rf_source_z_exploded_vec_,
                                  rf_source_power_watts_exploded_vec_,
                                  rf_source_freq_mhz_exploded_vec_,
                                  rf_source_antenna_azimuth_degrees_exploded_vec_,
                                  rf_source_antenna_downtilt_degrees_exploded_vec_,
                                  rf_source_antenna_type_exploded_vec_);
  }

 private:
  std::vector<S> rf_source_id_exploded_vec_;
  std::vector<T1> rf_source_x_exploded_vec_;
  std::vector<T1> rf_source_y_exploded_vec_;
  std::vector<Z1> rf_source_z_exploded_vec_;
  std::vector<T1> rf_source_power_watts_exploded_vec_;
  std::vector<T1> rf_source_freq_mhz_exploded_vec_;
  std::vector<T1> rf_source_antenna_azimuth_degrees_exploded_vec_;
  std::vector<T1> rf_source_antenna_downtilt_degrees_exploded_vec_;
  std::vector<TextEncodingDict> rf_source_antenna_type_exploded_vec_;
};

// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager, 
   Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> z_meters, Column<T1> tx_power_watts,
   Column<T1> tx_freq_mhz, Column<T1> antenna_azimuth_degrees, Column<T1> antenna_downtilt_degrees, Column<TextEncodingDict> antenna_type> rf_sources, 
   Cursor<Column<T2> x, Column<T2> y, Column<Z2> ground_elevation_amsl_meters, Column<Z2> terrain_elevation_amsl_meters, Column<Z2> terrain_attenuation_dbm_per_meter> terrain_elevations, 
   Cursor<Column<TextEncodingDict> antenna_type, Column<T3> antenna_gain,
   Column<Array<T3>> antenna_horizontal_degrees, Column<Array<T3>> antenna_horizontal_attenuation,
   Column<Array<T3>> antenna_vertical_degrees, Column<Array<T3>> antenna_vertical_attenuation> antenna_patterns, 
   bool rf_source_z_is_relative_to_terrain | default=true,
   bool geographic_coords | default=true,
   double bin_dim_meters,
   double assumed_receiver_height_agl | default=2.0, 
   double max_ray_travel_meters | default=3000.0,
   int64_t initial_rays_per_source | default=360,
   double rays_per_bin_autosplit_threshold | default=1.5,
   double min_receiver_signal_strength_dbm | default=-120.0,
   double default_source_height_agl_meters | default=20.0, 
   double ray_step_bin_multiple | default=1.0,
   int64_t loop_grain_size | default=4) | filter_table_function_transpose=on -> 
   Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters, 
   Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm, 
   S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double], T3=[float, double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2, typename T3>
TEMPLATE_NOINLINE int32_t tf_rf_prop_max_signal__cpu_template(
    TableFunctionManager& mgr,
    const Column<S>& rf_source_id,
    const Column<T1>& rf_source_x,
    const Column<T1>& rf_source_y,
    const Column<Z1>& rf_source_z,
    const Column<T1>& rf_source_power_watts,
    const Column<T1>& rf_source_freq_mhz,
    const Column<T1>& rf_source_antenna_azimuth_degrees,
    const Column<T1>& rf_source_antenna_downtilt_degrees,
    const Column<TextEncodingDict>& rf_source_antenna_type,
    const Column<T2>& terrain_x,
    const Column<T2>& terrain_y,
    const Column<Z2>& ground_z,
    const Column<Z2>& terrain_z,
    const Column<Z2>& terrain_attenuation_dbm_per_meter,
    const Column<TextEncodingDict>& antenna_type,
    const Column<T3>& antenna_gain,
    const Column<Array<T3>>& antenna_horizontal_degrees,
    const Column<Array<T3>>& antenna_horizontal_attenuation,
    const Column<Array<T3>>& antenna_vertical_degrees,
    const Column<Array<T3>>& antenna_vertical_attenuation,
    const bool rf_source_z_is_relative_to_terrain,
    const bool geographic_coords,
    const double bin_dim_meters,
    const double assumed_receiver_height_agl,
    const double max_ray_travel_meters,
    const int64_t initial_rays_per_source,
    const double rays_per_bin_autosplit_threshold,
    const double min_receiver_signal_strength_dbm,
    const double assumed_source_height_above_ground,
    const double ray_step_bin_multiple,
    const int64_t loop_grain_size,
    Column<T1>& out_x,
    Column<T1>& out_y,
    Column<Z1>& out_max_z,
    Column<S>& out_strongest_rf_source_id,
    Column<T1>& out_max_rf_signal_strength_dbm) {
  auto timer = DEBUG_TIMER(__func__);

  if (!is_valid_tf_input(bin_dim_meters, 0.0, BoundsType::Min, IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("bin_dim_meters must be > 0");
  }
  if (!is_valid_tf_input(
          assumed_receiver_height_agl, 0.0, BoundsType::Min, IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("assumed_receiver_height_agl must be >= 0");
  }
  if (!is_valid_tf_input(
          max_ray_travel_meters, 0.0, BoundsType::Min, IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("max_ray_travel_meters must be > 0");
  }
  if (!is_valid_tf_input(assumed_source_height_above_ground,
                         0.0,
                         BoundsType::Min,
                         IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("assumed_source_height_above_ground must be >= 0");
  }
  if (!is_valid_tf_input(loop_grain_size,
                         static_cast<decltype(loop_grain_size)>(1),
                         BoundsType::Min,
                         IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("loop_grain_size must be >= 1");
  }
  const int64_t num_rf_sources{rf_source_id.size()};

  auto [rf_source_power_watts_min, rf_source_power_watts_max] =
      get_column_min_max(rf_source_power_watts);

  if (num_rf_sources > 0 && rf_source_power_watts_min <= 0.0) {
    return mgr.ERROR_MESSAGE("RF source power (watts) must be > 0");
  }
  auto [rf_source_freq_mhz_min, rf_source_freq_mhz_max] =
      get_column_min_max(rf_source_freq_mhz);
  if (num_rf_sources > 0 && rf_source_freq_mhz_min <= 0.0) {
    return mgr.ERROR_MESSAGE("RF frequency (MHz) must be > 0");
  }

  constexpr int64_t num_vertical_attenuation_bins = 4096;

  // Note: The following input values are unchecked:
  // rf_source_signal_strength_dbm and min_receiver_signal_strength_dbm have unbounded
  // ranges num_rays_per_source and ray_step_bin_multiple set automatic defaults if values
  // are <= 0

  // Create ColumnList wrapper of terrain columns to input into GeoRaster
  std::vector<Z2*> terrain_ptrs = {
      ground_z.ptr_, terrain_z.ptr_, terrain_attenuation_dbm_per_meter.ptr_};
  ColumnList<Z2> terrain_cols(
      reinterpret_cast<int8_t**>(terrain_ptrs.data()), 3, ground_z.size());

  std::vector<RasterAggType> raster_agg_types = {
      RasterAggType::MIN, RasterAggType::MAX, RasterAggType::AVG};

  GeoRaster<T1, Z1> geo_raster(terrain_x,
                               terrain_y,
                               terrain_cols,
                               raster_agg_types,
                               bin_dim_meters,
                               geographic_coords,
                               true);

  if (geo_raster.num_bins_ == 0) {
    return mgr.ERROR_MESSAGE("terrain bins are empty");
  }
  geo_raster.fill_bins_from_neighbors(2 /* fill_radius */,
                                      true /* fill_only_nulls*/,
                                      RasterAggType::GAUSS_AVG,
                                      geo_raster.z_cols_[0]);
  geo_raster.fill_bins_from_neighbors(2 /* fill_radius */,
                                      true /* fill_only_nulls*/,
                                      RasterAggType::GAUSS_AVG,
                                      geo_raster.z_cols_[1]);
  geo_raster.fill_bins_from_neighbors(2 /* fill_radius */,
                                      true /* fill_only_nulls*/,
                                      RasterAggType::GAUSS_AVG,
                                      geo_raster.z_cols_[2]);

  geo_raster.outputDenseColumns(mgr, out_x, out_y, out_max_z, 1);

  if (check_interrupt()) {
    return mgr.ERROR_MESSAGE("Interrupted after GeoRaster");
  }

  Column<Z1> binned_ground_z(geo_raster.z_cols_[0].data(), geo_raster.num_bins_);
  Column<Z1> binned_terrain_attenuation_dbm_per_meter(geo_raster.z_cols_[2].data(),
                                                      geo_raster.num_bins_);

  // Pull these variables out of the geo_raster struct for convenience and perhaps
  // (slight) performance advantage
  const int64_t num_bins = geo_raster.num_bins_;
  const int64_t num_x_bins = geo_raster.num_x_bins_;
  const int64_t num_y_bins = geo_raster.num_y_bins_;

  // stash values in the key/value store
  geo_raster.setMetadata(mgr);

  const T1 numeric_min_sentinel{std::numeric_limits<T1>::lowest()};
  const S integer_null_sentinel{std::numeric_limits<S>::lowest()};

  // If no sources exist, fill with inline null (T::min())
  // in our system, instead of the sentinel (T::lowest()),
  // and return early

  bool all_sources_out_of_bounds = true;
  for (int64_t rf_source_idx = 0; rf_source_idx < num_rf_sources; ++rf_source_idx) {
    if (rf_source_x[rf_source_idx] >= geo_raster.x_min_ ||
        rf_source_x[rf_source_idx] <= geo_raster.x_max_ ||
        rf_source_y[rf_source_idx] >= geo_raster.y_min_ ||
        rf_source_y[rf_source_idx] <= geo_raster.y_max_) {
      // RF Source is in bounds
      all_sources_out_of_bounds = false;
      break;
    }
  }

  const T1 rf_signal_sentinel_fill_value =
      num_rf_sources == 0 || all_sources_out_of_bounds ? inline_null_value<T1>()
                                                       : numeric_min_sentinel;
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_bins),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      for (auto bin_idx = start_idx; bin_idx != end_idx; ++bin_idx) {
                        out_strongest_rf_source_id[bin_idx] = integer_null_sentinel;
                        out_max_rf_signal_strength_dbm[bin_idx] =
                            rf_signal_sentinel_fill_value;
                      }
                    });

  if (num_rf_sources == 0 || all_sources_out_of_bounds) {
    return num_bins;
  }

  std::vector<double> rf_source_power_dbm(num_rf_sources);
  std::transform(rf_source_power_watts.ptr_,
                 rf_source_power_watts.ptr_ + num_rf_sources,
                 rf_source_power_dbm.data(),
                 convert_power_watts_to_dbm<T1>);

  try {
    const auto antenna_pattern_map =
        generate_antenna_pattern_map(rf_source_antenna_type,
                                     antenna_type,
                                     antenna_gain,
                                     antenna_horizontal_degrees,
                                     antenna_horizontal_attenuation,
                                     antenna_vertical_degrees,
                                     antenna_vertical_attenuation);

    double max_ray_travel_meters_for_min_signal = 0.0;
    for (int64_t rf_source_idx = 0; rf_source_idx < num_rf_sources; ++rf_source_idx) {
      max_ray_travel_meters_for_min_signal =
          std::max(max_ray_travel_meters_for_min_signal,
                   get_min_distance_for_power_frequency_antenna_max_gain(
                       min_receiver_signal_strength_dbm,
                       rf_source_power_dbm[rf_source_idx],
                       rf_source_freq_mhz[rf_source_idx],
                       antenna_pattern_map[rf_source_idx].gain +
                           antenna_pattern_map[rf_source_idx].max_horizontal_gain +
                           antenna_pattern_map[rf_source_idx].max_vertical_gain));
    }

    const double max_ray_travel_meters_calculated =
        std::min(max_ray_travel_meters, max_ray_travel_meters_for_min_signal);

    if (max_ray_travel_meters_calculated <= 0.0) {
      return mgr.ERROR_MESSAGE("No RF sources have sufficient power");
    }

    const RasterOffsets<T1> raster_offsets(max_ray_travel_meters_calculated,
                                           bin_dim_meters,
                                           true /* -> only_first_quarter_radian */);
    const T1 ray_step_in_bins =
        (ray_step_bin_multiple > 0.0 &&
                 ray_step_bin_multiple < max_ray_travel_meters_calculated
             ? ray_step_bin_multiple
             : 1.0);

    const int32_t num_rays =
        initial_rays_per_source > 0 && initial_rays_per_source < 10000
            ? initial_rays_per_source
            : 64;
    const size_t tbb_loop_grain_size = static_cast<int64_t>(loop_grain_size);

    const int32_t partition_size = max_ray_travel_meters_calculated / bin_dim_meters;
    const int32_t num_x_partitions =
        num_x_bins / partition_size +
        static_cast<int32_t>((num_x_bins % partition_size) > 0);
    const int32_t num_y_partitions =
        num_y_bins / partition_size +
        static_cast<int32_t>((num_y_bins % partition_size) > 0);
    const int32_t num_partitions = num_x_partitions * num_y_partitions;
    const int32_t num_local_bins_per_dim = partition_size * 3;
    const int32_t num_local_bins = num_local_bins_per_dim * num_local_bins_per_dim;

    const PartitionInfo partition_info =
        get_partition_info<T1>(rf_source_x,
                               rf_source_y,
                               geo_raster.x_min_,
                               geo_raster.x_scale_input_to_bin_,
                               geo_raster.y_min_,
                               geo_raster.y_scale_input_to_bin_,
                               partition_size,
                               num_x_partitions,
                               num_y_partitions);

    std::vector<std::mutex> spatial_output_mutexes(num_partitions);
    // constexpr int64_t num_vertical_angles{720};

    if (check_interrupt()) {
      throw std::runtime_error("Interrupted before main thread launch");
    }

    tbb::parallel_for(
        tbb::blocked_range<int64_t>(
            partition_info.first_valid_element, num_rf_sources, tbb_loop_grain_size),
        [&](const tbb::blocked_range<int64_t>& r) {
          std::vector<T1> local_max_signal_dbms(num_local_bins, numeric_min_sentinel);
          std::vector<S> local_max_signal_dbms_source_ids(num_local_bins,
                                                          integer_null_sentinel);

          int32_t old_centroid_partition = std::numeric_limits<int32_t>::lowest();
          size_t current_local_partition_num_iterations = 0;

          for (int64_t source_idx = r.begin(); source_idx != r.end(); ++source_idx) {
            const int64_t permuted_idx = partition_info.get_permuted_idx(source_idx);
            const T1 source_x = rf_source_x[permuted_idx];
            const T1 source_y = rf_source_y[permuted_idx];
            const S source_id = rf_source_id[permuted_idx];
            const T1 source_power_dbm = rf_source_power_dbm[permuted_idx];
            const T1 source_freq_mhz = rf_source_freq_mhz[permuted_idx];
            const T1 source_antenna_azimuth_degrees =
                rf_source_antenna_azimuth_degrees[permuted_idx];
            const T1 source_antenna_downtilt_degrees =
                rf_source_antenna_downtilt_degrees[permuted_idx];
            const int32_t source_x_bin =
                (source_x - geo_raster.x_min_) * geo_raster.x_scale_input_to_bin_;
            const int32_t source_y_bin =
                (source_y - geo_raster.y_min_) * geo_raster.y_scale_input_to_bin_;
            if (geo_raster.is_bin_out_of_bounds(source_x_bin, source_y_bin)) {
              continue;
            }
            const Z1 source_z_input = rf_source_z[permuted_idx];
            const Z1 source_z = rf_source_z_is_relative_to_terrain
                                    ? geo_raster.offset_source_z_from_raster_z(
                                          source_x_bin, source_y_bin, source_z_input, 1)
                                    : source_z_input;
            if (geo_raster.is_null(source_z)) {
              continue;
            }

            if (check_interrupt()) {
              throw std::runtime_error("Interrupted in source loop");
            }

            const int32_t x_centroid_partition = source_x_bin / partition_size;
            const int32_t y_centroid_partition = source_y_bin / partition_size;
            const int32_t centroid_partition = x_y_bin_to_bin_index(
                x_centroid_partition, y_centroid_partition, num_x_partitions);
            if (centroid_partition != old_centroid_partition &&
                old_centroid_partition != std::numeric_limits<int32_t>::lowest()) {
              const auto old_partition_x_y_indexes =
                  bin_to_x_y_bin_indexes(old_centroid_partition, num_x_partitions);
              write_local_signal_dbm_output(out_max_rf_signal_strength_dbm,
                                            out_strongest_rf_source_id,
                                            local_max_signal_dbms,
                                            local_max_signal_dbms_source_ids,
                                            old_partition_x_y_indexes,
                                            partition_size,
                                            num_x_partitions,
                                            num_y_partitions,
                                            num_x_bins,
                                            num_y_bins,
                                            numeric_min_sentinel,
                                            spatial_output_mutexes);
              current_local_partition_num_iterations = 0;
              std::fill(local_max_signal_dbms.begin(),
                        local_max_signal_dbms.end(),
                        numeric_min_sentinel);
              std::fill(local_max_signal_dbms_source_ids.begin(),
                        local_max_signal_dbms_source_ids.end(),
                        integer_null_sentinel);
            }
            old_centroid_partition = centroid_partition;
            current_local_partition_num_iterations++;

            const int32_t global_x_bin_offset =
                (x_centroid_partition - 1) * partition_size;
            const int32_t global_y_bin_offset =
                (y_centroid_partition - 1) * partition_size;
            const T1 start_x_bin = source_x_bin + 0.5;  // convert to fp up front
            const T1 start_y_bin = source_y_bin + 0.5;  // convert to fp up front
            const T1 angle_step = 2.0 * M_PI / num_rays;
            const T1 ray_step_meters = ray_step_in_bins * bin_dim_meters;
            const double this_source_max_ray_travel_meters_calculated =
                std::min(get_min_distance_for_power_frequency_antenna_max_gain(
                             min_receiver_signal_strength_dbm,
                             source_power_dbm,
                             source_freq_mhz,
                             antenna_pattern_map[permuted_idx].max_horizontal_gain),
                         max_ray_travel_meters);
            const int32_t num_ray_steps =
                this_source_max_ray_travel_meters_calculated / ray_step_meters;

            for (int32_t r = 0; r < num_rays; ++r) {
              if (check_interrupt()) {
                throw std::runtime_error("Interrupted in ray loop");
              }

              const T1 angle_radians = r * angle_step;
              AngleAttenuationsFlatMap<T1> vertical_angle_attenuations(
                  -M_PI * 0.5, M_PI * 0.5, num_vertical_attenuation_bins);
              propagate_ray_with_attenuation(
                  source_id,
                  start_x_bin,
                  start_y_bin,
                  source_z,
                  source_power_dbm,
                  source_freq_mhz,
                  source_antenna_azimuth_degrees,
                  source_antenna_downtilt_degrees,
                  antenna_pattern_map[permuted_idx],
                  global_x_bin_offset,
                  global_y_bin_offset,
                  assumed_receiver_height_agl,
                  source_z - assumed_source_height_above_ground,
                  source_z - assumed_source_height_above_ground,
                  angle_radians,
                  angle_step,
                  static_cast<T1>(rays_per_bin_autosplit_threshold),
                  0,
                  num_ray_steps,
                  numeric_min_sentinel,  // starting_max_obscured_z_slope
                  ray_step_in_bins,
                  num_local_bins_per_dim,
                  static_cast<T1>(min_receiver_signal_strength_dbm),
                  binned_ground_z,
                  out_max_z,
                  binned_terrain_attenuation_dbm_per_meter,
                  geo_raster,
                  vertical_angle_attenuations,
                  local_max_signal_dbms,
                  local_max_signal_dbms_source_ids);
            }
          }
          if (current_local_partition_num_iterations > 0) {
            const auto old_partition_x_y_indexes =
                bin_to_x_y_bin_indexes(old_centroid_partition, num_x_partitions);
            write_local_signal_dbm_output(out_max_rf_signal_strength_dbm,
                                          out_strongest_rf_source_id,
                                          local_max_signal_dbms,
                                          local_max_signal_dbms_source_ids,
                                          old_partition_x_y_indexes,
                                          partition_size,
                                          num_x_partitions,
                                          num_y_partitions,
                                          num_x_bins,
                                          num_y_bins,
                                          numeric_min_sentinel,
                                          spatial_output_mutexes);
          }
        });
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_bins),
        [&](const tbb::blocked_range<size_t>& r) {
          for (size_t bin_idx = r.begin(); bin_idx != r.end(); ++bin_idx) {
            if (out_max_rf_signal_strength_dbm[bin_idx] == numeric_min_sentinel) {
              out_max_rf_signal_strength_dbm.setNull(bin_idx);
            }
          }
        });
    return num_bins;
  } catch (std::exception& e) {
    // We need to catch any exceptions and throw
    // In particular, exceptions can be thrown
    // building the antenna pattern map
    return mgr.ERROR_MESSAGE(e.what());
  }
}

// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager, 
   Cursor<Column<Array<S>> rf_source_id, Column<T1> x, Column<T1> y,
   Column<Z1> z_meters, Column<Array<T1>> tx_power_watts, Column<Array<T1>> tx_freq_mhz,
   Column<Array<T1>> antenna_azimuth_degrees, Column<Array<T1>> antenna_downtilt_degrees,
   Column<Array<TextEncodingDict>> antenna_type> rf_sources, 
   Cursor<Column<T2> x, Column<T2> y, Column<Z2> ground_elevation_amsl_meters, Column<Z2> terrain_elevation_amsl_meters, Column<Z2> terrain_attenuation_dbm_per_meter> terrain_elevations, 
   Cursor<Column<TextEncodingDict> antenna_type, Column<T3> antenna_gain,
   Column<Array<T3>> antenna_horizontal_degrees, Column<Array<T3>> antenna_horizontal_attenuation,
   Column<Array<T3>> antenna_vertical_degrees, Column<Array<T3>> vertical_attenuation> antenna_patterns, 
   bool rf_source_z_is_relative_to_terrain, bool geographic_coords, double bin_dim_meters, double assumed_receiver_height_agl,
   double max_ray_travel_meters, int64_t initial_rays_per_source, double rays_per_bin_autosplit_threshold,
   double min_receiver_signal_strength_dbm, double default_source_height_agl_meters, 
   double ray_step_bin_multiple, int64_t loop_grain_size) | filter_table_function_transpose=on -> 
   Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters, 
   Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm, 
   S=[int32_t, int64_t, TextEncodingDict], T1=[float, double], Z1=[double], T2=[double], Z2=[double], T3=[float, double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2, typename T3>
TEMPLATE_NOINLINE int32_t tf_rf_prop_max_signal__cpu_template(
    TableFunctionManager& mgr,
    const Column<Array<S>>& rf_source_id,
    const Column<T1>& rf_source_x,
    const Column<T1>& rf_source_y,
    const Column<Z1>& rf_source_z,
    const Column<Array<T1>>& rf_source_power_watts,
    const Column<Array<T1>>& rf_source_freq_mhz,
    const Column<Array<T1>>& rf_source_antenna_azimuth_degrees,
    const Column<Array<T1>>& rf_source_antenna_downtilt_degrees,
    const Column<Array<TextEncodingDict>>& rf_source_antenna_type,
    const Column<T2>& terrain_x,
    const Column<T2>& terrain_y,
    const Column<Z2>& ground_z,
    const Column<Z2>& terrain_z,
    const Column<Z2>& terrain_attenuation_dbm_per_meter,
    const Column<TextEncodingDict>& antenna_type,
    const Column<T3>& antenna_gain,
    const Column<Array<T3>>& antenna_horizontal_degrees,
    const Column<Array<T3>>& antenna_horizontal_attenuation,
    const Column<Array<T3>>& antenna_vertical_degrees,
    const Column<Array<T3>>& antenna_vertical_attenuation,
    const bool rf_source_z_is_relative_to_terrain,
    const bool geographic_coords,
    const double bin_dim_meters,
    const double assumed_receiver_height_agl,
    const double max_ray_travel_meters,
    const int64_t initial_rays_per_source,
    const double rays_per_bin_autosplit_threshold,
    const double min_receiver_signal_strength_dbm,
    const double assumed_source_height_above_ground,
    const double ray_step_bin_multiple,
    const int64_t loop_grain_size,
    Column<T1>& out_x,
    Column<T1>& out_y,
    Column<Z1>& out_max_z,
    Column<S>& out_strongest_rf_source_id,
    Column<T1>& out_max_rf_signal_strength_dbm) {
  try {
    AntennaArrayFlattener antenna_array_flattener(rf_source_id,
                                                  rf_source_x,
                                                  rf_source_y,
                                                  rf_source_z,
                                                  rf_source_power_watts,
                                                  rf_source_freq_mhz,
                                                  rf_source_antenna_azimuth_degrees,
                                                  rf_source_antenna_downtilt_degrees,
                                                  rf_source_antenna_type);
    auto flattened_antenna_params = antenna_array_flattener.getFlattenedAntennaParams(
        mgr, rf_source_id, rf_source_antenna_type);
    return tf_rf_prop_max_signal__cpu_template(
        mgr,
        flattened_antenna_params.rf_source_id_exploded,
        flattened_antenna_params.rf_source_x_exploded,
        flattened_antenna_params.rf_source_y_exploded,
        flattened_antenna_params.rf_source_z_exploded,
        flattened_antenna_params.rf_source_power_watts_exploded,
        flattened_antenna_params.rf_source_freq_mhz_exploded,
        flattened_antenna_params.rf_source_antenna_azimuth_degrees_exploded,
        flattened_antenna_params.rf_source_antenna_downtilt_degrees_exploded,
        flattened_antenna_params.rf_source_antenna_type_exploded,
        terrain_x,
        terrain_y,
        ground_z,
        terrain_z,
        terrain_attenuation_dbm_per_meter,
        antenna_type,
        antenna_gain,
        antenna_horizontal_degrees,
        antenna_horizontal_attenuation,
        antenna_vertical_degrees,
        antenna_vertical_attenuation,
        rf_source_z_is_relative_to_terrain,
        geographic_coords,
        bin_dim_meters,
        assumed_receiver_height_agl,
        max_ray_travel_meters,
        initial_rays_per_source,
        rays_per_bin_autosplit_threshold,
        min_receiver_signal_strength_dbm,
        assumed_source_height_above_ground,
        ray_step_bin_multiple,
        loop_grain_size,
        out_x,
        out_y,
        out_max_z,
        out_strongest_rf_source_id,
        out_max_rf_signal_strength_dbm);

  } catch (std::exception& e) {
    // We need to catch any exceptions and throw
    // In particular, exceptions can be thrown
    // building the antenna pattern map
    return mgr.ERROR_MESSAGE(e.what());
  }
}

// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager, 
   Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> z_meters, Column<T1> tx_power_watts,
   Column<T1> tx_freq_mhz, Column<T1> antenna_azimuth_degrees, Column<T1> antenna_downtilt_degrees, Column<TextEncodingDict> antenna_type> rf_sources, 
   Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations, 
   Cursor<Column<TextEncodingDict> antenna_type, Column<T3> antenna_gain,
   Column<Array<T3>> antenna_horizontal_degrees, Column<Array<T3>> antenna_horizontal_attenuation,
   Column<Array<T3>> antenna_vertical_degrees, Column<Array<T3>> antenna_vertical_attenuation> antenna_patterns, 
   bool rf_source_z_is_relative_to_terrain | default = true,
   bool geographic_coords | default = true,
   double bin_dim_meters,
   double assumed_receiver_height_agl | default = 2.0, 
   double max_ray_travel_meters | default = 3000.0,
   int64_t initial_rays_per_source | default = 360,
   double rays_per_bin_autosplit_threshold | default = 1.5,
   double min_receiver_signal_strength_dbm | default = -120.0,
   double default_source_height_agl_meters | default = 20.0, 
   double ray_step_bin_multiple | default = 1.0,
   int64_t loop_grain_size | default = 4) | filter_table_function_transpose=on -> 
   Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters, 
   Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm, 
   S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double], T3=[float, double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2, typename T3>
TEMPLATE_NOINLINE int32_t tf_rf_prop_max_signal__cpu_template(
    TableFunctionManager& mgr,
    const Column<S>& rf_source_id,
    const Column<T1>& rf_source_x,
    const Column<T1>& rf_source_y,
    const Column<Z1>& rf_source_z,
    const Column<T1>& rf_source_power_watts,
    const Column<T1>& rf_source_freq_mhz,
    const Column<T1>& rf_source_antenna_azimuth_degrees,
    const Column<T1>& rf_source_antenna_downtilt_degrees,
    const Column<TextEncodingDict>& rf_source_antenna_type,
    const Column<T2>& terrain_x,
    const Column<T2>& terrain_y,
    const Column<Z2>& terrain_z,
    const Column<TextEncodingDict>& antenna_type,
    const Column<T3>& antenna_gain,
    const Column<Array<T3>>& antenna_horizontal_degrees,
    const Column<Array<T3>>& antenna_horizontal_attenuation,
    const Column<Array<T3>>& antenna_vertical_degrees,
    const Column<Array<T3>>& antenna_vertical_attenuation,
    const bool rf_source_z_is_relative_to_terrain,
    const bool geographic_coords,
    const double bin_dim_meters,
    const double assumed_receiver_height_agl,
    const double max_ray_travel_meters,
    const int64_t initial_rays_per_source,
    const double rays_per_bin_autosplit_threshold,
    const double min_receiver_signal_strength_dbm,
    const double assumed_source_height_above_ground,
    const double ray_step_bin_multiple,
    const int64_t loop_grain_size,
    Column<T1>& out_x,
    Column<T1>& out_y,
    Column<Z1>& out_max_z,
    Column<S>& out_strongest_rf_source_id,
    Column<T1>& out_max_rf_signal_strength_dbm) {
  auto timer = DEBUG_TIMER(__func__);
  if (!is_valid_tf_input(bin_dim_meters, 0.0, BoundsType::Min, IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("bin_dim_meters must be > 0");
  }
  if (!is_valid_tf_input(
          assumed_receiver_height_agl, 0.0, BoundsType::Min, IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("assumed_receiver_height_agl must be >= 0");
  }
  if (!is_valid_tf_input(
          max_ray_travel_meters, 0.0, BoundsType::Min, IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("max_ray_travel_meters must be > 0");
  }
  if (!is_valid_tf_input(assumed_source_height_above_ground,
                         0.0,
                         BoundsType::Min,
                         IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("assumed_source_height_above_ground must be >= 0");
  }
  if (!is_valid_tf_input(loop_grain_size,
                         static_cast<decltype(loop_grain_size)>(1),
                         BoundsType::Min,
                         IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("loop_grain_size must be >= 1");
  }
  const int64_t num_rf_sources{rf_source_id.size()};

  auto [rf_source_power_watts_min, rf_source_power_watts_max] =
      get_column_min_max(rf_source_power_watts);
  if (num_rf_sources > 0 && rf_source_power_watts_min <= 0.0) {
    return mgr.ERROR_MESSAGE("RF source power (watts) must be > 0");
  }
  auto [rf_source_freq_mhz_min, rf_source_freq_mhz_max] =
      get_column_min_max(rf_source_freq_mhz);
  if (num_rf_sources > 0 && rf_source_freq_mhz_min <= 0.0) {
    return mgr.ERROR_MESSAGE("RF frequency (MHz) must be > 0");
  }

  // Note: The following input values are unchecked:
  // rf_source_signal_strength_dbm and min_receiver_signal_strength_dbm have unbounded
  // ranges num_rays_per_source and ray_step_bin_multiple set automatic defaults if values
  // are <= 0

  GeoRaster<T1, Z1> geo_raster(terrain_x,
                               terrain_y,
                               terrain_z,
                               RasterAggType::MAX,
                               bin_dim_meters,
                               geographic_coords,
                               true);

  if (geo_raster.num_bins_ == 0) {
    return mgr.ERROR_MESSAGE("terrain bins are empty");
  }
  geo_raster.fill_bins_from_neighbors(2 /* fill_radius */, true /* fill_only_nulls*/);

  geo_raster.outputDenseColumns(mgr, out_x, out_y, out_max_z, 1 /* terrain_z index */);

  if (check_interrupt()) {
    return mgr.ERROR_MESSAGE("Interrupted after GeoRaster");
  }

  // Pull these variables out of the geo_raster struct for convenience and perhaps
  // (slight) performance advantage
  const int64_t num_bins = geo_raster.num_bins_;
  const int64_t num_x_bins = geo_raster.num_x_bins_;
  const int64_t num_y_bins = geo_raster.num_y_bins_;

  // stash values in the key/value store
  geo_raster.setMetadata(mgr);

  const T1 numeric_min_sentinel{std::numeric_limits<T1>::lowest()};
  const S integer_null_sentinel{std::numeric_limits<S>::lowest()};

  // If no sources exist, fill with inline null (T::min())
  // in our system, instead of the sentinel (T::lowest()),
  // and return early

  bool all_sources_out_of_bounds = true;
  for (int64_t rf_source_idx = 0; rf_source_idx < num_rf_sources; ++rf_source_idx) {
    if (rf_source_x[rf_source_idx] >= geo_raster.x_min_ ||
        rf_source_x[rf_source_idx] <= geo_raster.x_max_ ||
        rf_source_y[rf_source_idx] >= geo_raster.y_min_ ||
        rf_source_y[rf_source_idx] <= geo_raster.y_max_) {
      // RF Source is in bounds
      all_sources_out_of_bounds = false;
      break;
    }
  }

  const T1 rf_signal_sentinel_fill_value =
      num_rf_sources == 0 || all_sources_out_of_bounds ? inline_null_value<T1>()
                                                       : numeric_min_sentinel;
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_bins),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      for (auto bin_idx = start_idx; bin_idx != end_idx; ++bin_idx) {
                        out_strongest_rf_source_id[bin_idx] = integer_null_sentinel;
                        out_max_rf_signal_strength_dbm[bin_idx] =
                            rf_signal_sentinel_fill_value;
                      }
                    });

  if (num_rf_sources == 0 || all_sources_out_of_bounds) {
    return num_bins;
  }

  std::vector<double> rf_source_power_dbm(num_rf_sources);
  std::transform(rf_source_power_watts.ptr_,
                 rf_source_power_watts.ptr_ + num_rf_sources,
                 rf_source_power_dbm.data(),
                 convert_power_watts_to_dbm<T1>);

  try {
    const auto antenna_pattern_map =
        generate_antenna_pattern_map(rf_source_antenna_type,
                                     antenna_type,
                                     antenna_gain,
                                     antenna_horizontal_degrees,
                                     antenna_horizontal_attenuation,
                                     antenna_vertical_degrees,
                                     antenna_vertical_attenuation);

    double max_ray_travel_meters_for_min_signal = 0.0;
    for (int64_t rf_source_idx = 0; rf_source_idx < num_rf_sources; ++rf_source_idx) {
      max_ray_travel_meters_for_min_signal =
          std::max(max_ray_travel_meters_for_min_signal,
                   get_min_distance_for_power_frequency_antenna_max_gain(
                       min_receiver_signal_strength_dbm,
                       rf_source_power_dbm[rf_source_idx],
                       rf_source_freq_mhz[rf_source_idx],
                       antenna_pattern_map[rf_source_idx].gain +
                           antenna_pattern_map[rf_source_idx].max_horizontal_gain +
                           antenna_pattern_map[rf_source_idx].max_vertical_gain));
    }

    const double max_ray_travel_meters_calculated =
        std::min(max_ray_travel_meters, max_ray_travel_meters_for_min_signal);

    if (max_ray_travel_meters_calculated <= 0.0) {
      return mgr.ERROR_MESSAGE("No RF sources have sufficient power");
    }

    const RasterOffsets<T1> raster_offsets(max_ray_travel_meters_calculated,
                                           bin_dim_meters,
                                           true /* -> only_first_quarter_radian */);
    const T1 ray_step_in_bins =
        (ray_step_bin_multiple > 0.0 &&
                 ray_step_bin_multiple < max_ray_travel_meters_calculated
             ? ray_step_bin_multiple
             : 1.0);

    const int32_t num_rays =
        initial_rays_per_source > 0 && initial_rays_per_source < 10000
            ? initial_rays_per_source
            : 64;
    const size_t tbb_loop_grain_size = static_cast<int64_t>(loop_grain_size);

    const int32_t partition_size = max_ray_travel_meters_calculated / bin_dim_meters;
    const int32_t num_x_partitions =
        num_x_bins / partition_size +
        static_cast<int32_t>((num_x_bins % partition_size) > 0);
    const int32_t num_y_partitions =
        num_y_bins / partition_size +
        static_cast<int32_t>((num_y_bins % partition_size) > 0);
    const int32_t num_partitions = num_x_partitions * num_y_partitions;
    const int32_t num_local_bins_per_dim = partition_size * 3;
    const int32_t num_local_bins = num_local_bins_per_dim * num_local_bins_per_dim;

    const PartitionInfo partition_info =
        get_partition_info<T1>(rf_source_x,
                               rf_source_y,
                               geo_raster.x_min_,
                               geo_raster.x_scale_input_to_bin_,
                               geo_raster.y_min_,
                               geo_raster.y_scale_input_to_bin_,
                               partition_size,
                               num_x_partitions,
                               num_y_partitions);

    std::vector<std::mutex> spatial_output_mutexes(num_partitions);

    if (check_interrupt()) {
      throw std::runtime_error("Interrupted before main thread launch");
    }

    tbb::parallel_for(
        tbb::blocked_range<int64_t>(
            partition_info.first_valid_element, num_rf_sources, tbb_loop_grain_size),
        [&](const tbb::blocked_range<int64_t>& r) {
          std::vector<T1> local_max_signal_dbms(num_local_bins, numeric_min_sentinel);
          std::vector<S> local_max_signal_dbms_source_ids(num_local_bins,
                                                          integer_null_sentinel);

          int32_t old_centroid_partition = std::numeric_limits<int32_t>::lowest();
          size_t current_local_partition_num_iterations = 0;

          for (int64_t source_idx = r.begin(); source_idx != r.end(); ++source_idx) {
            const int64_t permuted_idx = partition_info.get_permuted_idx(source_idx);
            const T1 source_x = rf_source_x[permuted_idx];
            const T1 source_y = rf_source_y[permuted_idx];
            const S source_id = rf_source_id[permuted_idx];
            const T1 source_power_dbm = rf_source_power_dbm[permuted_idx];
            const T1 source_freq_mhz = rf_source_freq_mhz[permuted_idx];
            const T1 source_antenna_azimuth_degrees =
                rf_source_antenna_azimuth_degrees[permuted_idx];
            const T1 source_antenna_downtilt_degrees =
                rf_source_antenna_downtilt_degrees[permuted_idx];
            const int32_t source_x_bin =
                (source_x - geo_raster.x_min_) * geo_raster.x_scale_input_to_bin_;
            const int32_t source_y_bin =
                (source_y - geo_raster.y_min_) * geo_raster.y_scale_input_to_bin_;
            if (geo_raster.is_bin_out_of_bounds(source_x_bin, source_y_bin)) {
              continue;
            }
            const Z1 source_z_input = rf_source_z[permuted_idx];
            const Z1 source_z = rf_source_z_is_relative_to_terrain
                                    ? geo_raster.offset_source_z_from_raster_z(
                                          source_x_bin, source_y_bin, source_z_input)
                                    : source_z_input;
            if (geo_raster.is_null(source_z)) {
              continue;
            }

            if (check_interrupt()) {
              throw std::runtime_error("Interrupted in source loop");
            }

            const int32_t x_centroid_partition = source_x_bin / partition_size;
            const int32_t y_centroid_partition = source_y_bin / partition_size;
            const int32_t centroid_partition = x_y_bin_to_bin_index(
                x_centroid_partition, y_centroid_partition, num_x_partitions);
            if (centroid_partition != old_centroid_partition &&
                old_centroid_partition != std::numeric_limits<int32_t>::lowest()) {
              const auto old_partition_x_y_indexes =
                  bin_to_x_y_bin_indexes(old_centroid_partition, num_x_partitions);
              write_local_signal_dbm_output(out_max_rf_signal_strength_dbm,
                                            out_strongest_rf_source_id,
                                            local_max_signal_dbms,
                                            local_max_signal_dbms_source_ids,
                                            old_partition_x_y_indexes,
                                            partition_size,
                                            num_x_partitions,
                                            num_y_partitions,
                                            num_x_bins,
                                            num_y_bins,
                                            numeric_min_sentinel,
                                            spatial_output_mutexes);
              current_local_partition_num_iterations = 0;
              std::fill(local_max_signal_dbms.begin(),
                        local_max_signal_dbms.end(),
                        numeric_min_sentinel);
              std::fill(local_max_signal_dbms_source_ids.begin(),
                        local_max_signal_dbms_source_ids.end(),
                        integer_null_sentinel);
            }
            old_centroid_partition = centroid_partition;
            current_local_partition_num_iterations++;

            const int32_t global_x_bin_offset =
                (x_centroid_partition - 1) * partition_size;
            const int32_t global_y_bin_offset =
                (y_centroid_partition - 1) * partition_size;
            const T1 start_x_bin = source_x_bin + 0.5;  // convert to fp up front
            const T1 start_y_bin = source_y_bin + 0.5;  // convert to fp up front
            const T1 angle_step = 2.0 * M_PI / num_rays;
            const T1 ray_step_meters = ray_step_in_bins * bin_dim_meters;
            const double this_source_max_ray_travel_meters_calculated =
                std::min(get_min_distance_for_power_frequency_antenna_max_gain(
                             min_receiver_signal_strength_dbm,
                             source_power_dbm,
                             source_freq_mhz,
                             antenna_pattern_map[permuted_idx].max_horizontal_gain),
                         max_ray_travel_meters);
            const int32_t num_ray_steps =
                this_source_max_ray_travel_meters_calculated / ray_step_meters;

            for (int32_t r = 0; r < num_rays; ++r) {
              if (check_interrupt()) {
                throw std::runtime_error("Interrupted in ray loop");
              }

              const T1 angle_radians = r * angle_step;
              propagate_ray(source_id,
                            start_x_bin,
                            start_y_bin,
                            source_z,
                            source_power_dbm,
                            source_freq_mhz,
                            source_antenna_azimuth_degrees,
                            source_antenna_downtilt_degrees,
                            antenna_pattern_map[permuted_idx],
                            global_x_bin_offset,
                            global_y_bin_offset,
                            assumed_receiver_height_agl,
                            source_z - assumed_source_height_above_ground,
                            angle_radians,
                            angle_step,
                            static_cast<T1>(rays_per_bin_autosplit_threshold),
                            0,
                            num_ray_steps,
                            numeric_min_sentinel,  // starting_max_obscured_z_slope
                            ray_step_in_bins,
                            num_local_bins_per_dim,
                            static_cast<T1>(min_receiver_signal_strength_dbm),
                            out_max_z,
                            geo_raster,
                            local_max_signal_dbms,
                            local_max_signal_dbms_source_ids);
            }
          }
          if (current_local_partition_num_iterations > 0) {
            const auto old_partition_x_y_indexes =
                bin_to_x_y_bin_indexes(old_centroid_partition, num_x_partitions);
            write_local_signal_dbm_output(out_max_rf_signal_strength_dbm,
                                          out_strongest_rf_source_id,
                                          local_max_signal_dbms,
                                          local_max_signal_dbms_source_ids,
                                          old_partition_x_y_indexes,
                                          partition_size,
                                          num_x_partitions,
                                          num_y_partitions,
                                          num_x_bins,
                                          num_y_bins,
                                          numeric_min_sentinel,
                                          spatial_output_mutexes);
          }
        });
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_bins),
        [&](const tbb::blocked_range<size_t>& r) {
          for (size_t bin_idx = r.begin(); bin_idx != r.end(); ++bin_idx) {
            if (out_max_rf_signal_strength_dbm[bin_idx] == numeric_min_sentinel) {
              out_max_rf_signal_strength_dbm.setNull(bin_idx);
            }
          }
        });
    return num_bins;
  } catch (std::exception& e) {
    // We need to catch any exceptions and throw
    // In particular, exceptions can be thrown
    // building the antenna pattern map
    return mgr.ERROR_MESSAGE(e.what());
  }
}

// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager, 
   Cursor<Column<Array<S>> rf_source_id, Column<T1> x, Column<T1> y,
   Column<Z1> z_meters, Column<Array<T1>> tx_power_watts, Column<Array<T1>> tx_freq_mhz,
   Column<Array<T1>> antenna_azimuth_degrees, Column<Array<T1>> antenna_downtilt_degrees,
   Column<Array<TextEncodingDict>> antenna_type> rf_sources, 
   Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations, 
   Cursor<Column<TextEncodingDict> antenna_type, Column<T3> antenna_gain,
   Column<Array<T3>> antenna_horizontal_degrees, Column<Array<T3>> antenna_horizontal_attenuation,
   Column<Array<T3>> antenna_vertical_degrees, Column<Array<T3>> vertical_attenuation> antenna_patterns, 
   bool rf_source_z_is_relative_to_terrain | default = true,
   bool geographic_coords | default = true,
   double bin_dim_meters,
   double assumed_receiver_height_agl | default = 2.0,
   double max_ray_travel_meters | default = 3000.0,
   int64_t initial_rays_per_source | default = 360,
   double rays_per_bin_autosplit_threshold | default = 1.5,
   double min_receiver_signal_strength_dbm | default = -120.0,
   double default_source_height_agl_meters | default = 20.0, 
   double ray_step_bin_multiple | default = 1.0,
   int64_t loop_grain_size | default = 4) | filter_table_function_transpose=on -> 
   Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters, 
   Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm, 
   S=[int32_t, int64_t, TextEncodingDict], T1=[float, double], Z1=[double], T2=[double], Z2=[double], T3=[float, double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2, typename T3>
TEMPLATE_NOINLINE int32_t tf_rf_prop_max_signal__cpu_template(
    TableFunctionManager& mgr,
    const Column<Array<S>>& rf_source_id,
    const Column<T1>& rf_source_x,
    const Column<T1>& rf_source_y,
    const Column<Z1>& rf_source_z,
    const Column<Array<T1>>& rf_source_power_watts,
    const Column<Array<T1>>& rf_source_freq_mhz,
    const Column<Array<T1>>& rf_source_antenna_azimuth_degrees,
    const Column<Array<T1>>& rf_source_antenna_downtilt_degrees,
    const Column<Array<TextEncodingDict>>& rf_source_antenna_type,
    const Column<T2>& terrain_x,
    const Column<T2>& terrain_y,
    const Column<Z2>& terrain_z,
    const Column<TextEncodingDict>& antenna_type,
    const Column<T3>& antenna_gain,
    const Column<Array<T3>>& antenna_horizontal_degrees,
    const Column<Array<T3>>& antenna_horizontal_attenuation,
    const Column<Array<T3>>& antenna_vertical_degrees,
    const Column<Array<T3>>& antenna_vertical_attenuation,
    const bool rf_source_z_is_relative_to_terrain,
    const bool geographic_coords,
    const double bin_dim_meters,
    const double assumed_receiver_height_agl,
    const double max_ray_travel_meters,
    const int64_t initial_rays_per_source,
    const double rays_per_bin_autosplit_threshold,
    const double min_receiver_signal_strength_dbm,
    const double assumed_source_height_above_ground,
    const double ray_step_bin_multiple,
    const int64_t loop_grain_size,
    Column<T1>& out_x,
    Column<T1>& out_y,
    Column<Z1>& out_max_z,
    Column<S>& out_strongest_rf_source_id,
    Column<T1>& out_max_rf_signal_strength_dbm) {
  try {
    AntennaArrayFlattener antenna_array_flattener(rf_source_id,
                                                  rf_source_x,
                                                  rf_source_y,
                                                  rf_source_z,
                                                  rf_source_power_watts,
                                                  rf_source_freq_mhz,
                                                  rf_source_antenna_azimuth_degrees,
                                                  rf_source_antenna_downtilt_degrees,
                                                  rf_source_antenna_type);
    auto flattened_antenna_params = antenna_array_flattener.getFlattenedAntennaParams(
        mgr, rf_source_id, rf_source_antenna_type);
    return tf_rf_prop_max_signal__cpu_template(
        mgr,
        flattened_antenna_params.rf_source_id_exploded,
        flattened_antenna_params.rf_source_x_exploded,
        flattened_antenna_params.rf_source_y_exploded,
        flattened_antenna_params.rf_source_z_exploded,
        flattened_antenna_params.rf_source_power_watts_exploded,
        flattened_antenna_params.rf_source_freq_mhz_exploded,
        flattened_antenna_params.rf_source_antenna_azimuth_degrees_exploded,
        flattened_antenna_params.rf_source_antenna_downtilt_degrees_exploded,
        flattened_antenna_params.rf_source_antenna_type_exploded,
        terrain_x,
        terrain_y,
        terrain_z,
        antenna_type,
        antenna_gain,
        antenna_horizontal_degrees,
        antenna_horizontal_attenuation,
        antenna_vertical_degrees,
        antenna_vertical_attenuation,
        rf_source_z_is_relative_to_terrain,
        geographic_coords,
        bin_dim_meters,
        assumed_receiver_height_agl,
        max_ray_travel_meters,
        initial_rays_per_source,
        rays_per_bin_autosplit_threshold,
        min_receiver_signal_strength_dbm,
        assumed_source_height_above_ground,
        ray_step_bin_multiple,
        loop_grain_size,
        out_x,
        out_y,
        out_max_z,
        out_strongest_rf_source_id,
        out_max_rf_signal_strength_dbm);

  } catch (std::exception& e) {
    // We need to catch any exceptions and throw
    // In particular, exceptions can be thrown
    // building the antenna pattern map
    return mgr.ERROR_MESSAGE(e.what());
  }
}

// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager, 
   Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> z_meters, Column<T1> tx_power_watts,
   Column<T1> tx_freq_mhz> rf_sources, 
   Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations, 
   bool rf_source_z_is_relative_to_terrain | default = true,
   bool geographic_coords | default = true,
   double bin_dim_meters,
   double max_ray_travel_meters | default = 3000.0,
   int64_t initial_rays_per_source | default = 360,
   double rays_per_bin_autosplit_threshold | default = 1.0,
   double min_receiver_signal_strength_dbm | default = -120.0,
   double default_source_height_agl_meters | default = 20.0, 
   double ray_step_bin_multiple | default = 1.0,
   int64_t loop_grain_size | default = 4) | filter_table_function_transpose=on -> 
   Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters, 
   Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm, 
   S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2>
TEMPLATE_NOINLINE int32_t
tf_rf_prop_max_signal__cpu_template(TableFunctionManager& mgr,
                                    const Column<S>& rf_source_id,
                                    const Column<T1>& rf_source_x,
                                    const Column<T1>& rf_source_y,
                                    const Column<Z1>& rf_source_z,
                                    const Column<T1>& rf_source_power_watts,
                                    const Column<T1>& rf_source_freq_mhz,
                                    const Column<T2>& terrain_x,
                                    const Column<T2>& terrain_y,
                                    const Column<Z2>& terrain_z,
                                    const bool rf_source_z_is_relative_to_terrain,
                                    const bool geographic_coords,
                                    const double bin_dim_meters,
                                    const double max_ray_travel_meters,
                                    const int64_t initial_rays_per_source,
                                    const double rays_per_bin_autosplit_threshold,
                                    const double min_receiver_signal_strength_dbm,
                                    const double assumed_source_height_above_ground,
                                    const double ray_step_bin_multiple,
                                    const int64_t loop_grain_size,
                                    Column<T1>& out_x,
                                    Column<T1>& out_y,
                                    Column<Z1>& out_max_z,
                                    Column<S>& out_strongest_rf_source_id,
                                    Column<T1>& out_max_rf_signal_strength_dbm) {
  const int64_t num_towers = rf_source_id.size();
  std::vector<T1> rf_source_antenna_azimuth_degrees_vec(num_towers, 0.0);
  std::vector<T1> rf_source_antenna_downtilt_degrees_vec(num_towers, 0.0);
  std::vector<TextEncodingDict> rf_source_antenna_type_vec(
      num_towers, std::numeric_limits<int32_t>::min());

  Column<T1> rf_source_antenna_azimuth_degrees(rf_source_antenna_azimuth_degrees_vec);
  Column<T1> rf_source_antenna_downtilt_degrees(rf_source_antenna_downtilt_degrees_vec);
  Column<TextEncodingDict> rf_source_antenna_type(rf_source_antenna_type_vec);

  std::vector<TextEncodingDict> antenna_type_vec;
  std::vector<float> antenna_gain_vec;
  Column<TextEncodingDict> antenna_type(antenna_type_vec);
  Column<float> antenna_gain(antenna_gain_vec);
  Column<Array<float>> antenna_horizontal_degrees(nullptr, 0);
  Column<Array<float>> antenna_horizontal_attenuation(nullptr, 0);
  Column<Array<float>> antenna_vertical_degrees(nullptr, 0);
  Column<Array<float>> antenna_vertical_attenuation(nullptr, 0);

  const double assumed_receiver_height_agl{0.0};

  return tf_rf_prop_max_signal__cpu_template(mgr,
                                             rf_source_id,
                                             rf_source_x,
                                             rf_source_y,
                                             rf_source_z,
                                             rf_source_power_watts,
                                             rf_source_freq_mhz,
                                             rf_source_antenna_azimuth_degrees,
                                             rf_source_antenna_downtilt_degrees,
                                             rf_source_antenna_type,
                                             terrain_x,
                                             terrain_y,
                                             terrain_z,
                                             antenna_type,
                                             antenna_gain,
                                             antenna_horizontal_degrees,
                                             antenna_horizontal_attenuation,
                                             antenna_vertical_degrees,
                                             antenna_vertical_attenuation,
                                             rf_source_z_is_relative_to_terrain,
                                             geographic_coords,
                                             bin_dim_meters,
                                             assumed_receiver_height_agl,
                                             max_ray_travel_meters,
                                             initial_rays_per_source,
                                             rays_per_bin_autosplit_threshold,
                                             min_receiver_signal_strength_dbm,
                                             assumed_source_height_above_ground,
                                             ray_step_bin_multiple,
                                             loop_grain_size,
                                             out_x,
                                             out_y,
                                             out_max_z,
                                             out_strongest_rf_source_id,
                                             out_max_rf_signal_strength_dbm);
}

// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager, 
   Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> z_meters> rf_sources, 
   bool rf_source_z_is_relative_to_terrain | default = true, double rf_source_signal_strength_dbm,
   double rf_source_signal_frequency_mhz | default = 3900.0,
   Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations, 
   bool geographic_coords | default = true,
   double bin_dim_meters,
   double max_ray_travel_meters | default = 3000.0,
   int64_t num_rays_per_source | default = 360, 
   double min_receiver_signal_strength_dbm | default = -120.0,
   double default_source_height_agl_meters | default = 20.0, 
   double ray_step_bin_multiple | default = 1.0,
   int64_t loop_grain_size | default = 4) | filter_table_function_transpose=on -> 
   Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters, 
   Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm, 
   S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2>
TEMPLATE_NOINLINE int32_t
tf_rf_prop_max_signal__cpu_template(TableFunctionManager& mgr,
                                    const Column<S>& rf_source_id,
                                    const Column<T1>& rf_source_x,
                                    const Column<T1>& rf_source_y,
                                    const Column<Z1>& rf_source_z,
                                    const bool rf_source_z_is_relative_to_terrain,
                                    const double rf_source_signal_strength_dbm,
                                    const double rf_source_signal_frequency_mhz,
                                    const Column<T2>& terrain_x,
                                    const Column<T2>& terrain_y,
                                    const Column<Z2>& terrain_z,
                                    const bool geographic_coords,
                                    const double bin_dim_meters,
                                    const double max_ray_travel_meters,
                                    const int64_t num_rays_per_source,
                                    const double min_receiver_signal_strength_dbm,
                                    const double assumed_source_height_above_ground,
                                    const double ray_step_bin_multiple,
                                    const int64_t loop_grain_size,
                                    Column<T1>& out_x,
                                    Column<T1>& out_y,
                                    Column<Z1>& out_max_z,
                                    Column<S>& out_strongest_rf_source_id,
                                    Column<T1>& out_max_rf_signal_strength_dbm) {
  const int64_t num_towers = rf_source_id.size();
  std::vector<T1> rf_source_power_watts_vec(
      num_towers, convert_power_dbm_to_watts(rf_source_signal_strength_dbm));
  std::vector<T1> rf_source_freq_mhz_vec(num_towers, rf_source_signal_frequency_mhz);
  Column<T1> rf_source_power_watts(rf_source_power_watts_vec);
  Column<T1> rf_source_freq_mhz(rf_source_freq_mhz_vec);

  const double rays_per_bin_autosplit_threshold = 1.5;

  return tf_rf_prop_max_signal__cpu_template(mgr,
                                             rf_source_id,
                                             rf_source_x,
                                             rf_source_y,
                                             rf_source_z,
                                             rf_source_power_watts,
                                             rf_source_freq_mhz,
                                             terrain_x,
                                             terrain_y,
                                             terrain_z,
                                             rf_source_z_is_relative_to_terrain,
                                             geographic_coords,
                                             bin_dim_meters,
                                             max_ray_travel_meters,
                                             num_rays_per_source,
                                             rays_per_bin_autosplit_threshold,
                                             min_receiver_signal_strength_dbm,
                                             assumed_source_height_above_ground,
                                             ray_step_bin_multiple,
                                             loop_grain_size,
                                             out_x,
                                             out_y,
                                             out_max_z,
                                             out_strongest_rf_source_id,
                                             out_max_rf_signal_strength_dbm);
}
// clang-format off
/*
  UDTF: tf_rf_prop_max_signal__cpu_template(TableFunctionManager,
  Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> z_meters> rf_sources,
  bool rf_source_z_is_relative_to_terrain | default = true, double rf_source_signal_strength_dbm,
  double rf_source_signal_frequency_mhz,
  Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations,
  double bin_dim_meters,
  double max_ray_travel_meters | default = 3000.0,
  int64_t num_rays_per_source | default = 360,
  double min_receiver_signal_strength_dbm | default = -120.0) | filter_table_function_transpose=on ->
  Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters,
  Column <S> rf_source_id | input_id=args<0>, Column<T1> max_rf_signal_strength_dbm,
  S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2>
TEMPLATE_NOINLINE int32_t
tf_rf_prop_max_signal__cpu_template(TableFunctionManager& mgr,
                                    const Column<S>& rf_source_id,
                                    const Column<T1>& rf_source_x,
                                    const Column<T1>& rf_source_y,
                                    const Column<Z1>& rf_source_z,
                                    const bool rf_source_z_is_relative_to_terrain,
                                    const double rf_source_signal_strength_dbm,
                                    const double rf_source_signal_frequency_mhz,
                                    const Column<T2>& terrain_x,
                                    const Column<T2>& terrain_y,
                                    const Column<Z2>& terrain_z,
                                    const double bin_dim_meters,
                                    const double max_ray_travel_meters,
                                    const int64_t num_rays_per_source,
                                    const double min_receiver_signal_strength_dbm,
                                    Column<T1>& out_x,
                                    Column<T1>& out_y,
                                    Column<Z1>& out_max_z,
                                    Column<S>& out_strongest_rf_source_id,
                                    Column<T1>& out_max_rf_signal_strength_dbm) {
  const bool geographic_coords =
      true;  // Terrain and tower data are in lon/lat (WGS-84). Bin size in degrees is
             // choosen based on first derivative of haversine distance at terrain data
             // centroid to make 1 bin width/height = bin_dim_meters
  const double assumed_source_height_above_ground =
      10.0;  // Only used for repeaters have no terrain elevation data at its base
  const double ray_step_bin_multiple =
      1.0;  // Sensible default, although slightly "fuller" results can be obtained with
            // steps < 1.0, at the expense of greater runtime
  const int64_t loop_grain_size =
      10;  // Sensible default derived via testing on many-cored hardware, mainly useful
           // for last-mile tuning

  return tf_rf_prop_max_signal__cpu_template(mgr,
                                             rf_source_id,
                                             rf_source_x,
                                             rf_source_y,
                                             rf_source_z,
                                             rf_source_z_is_relative_to_terrain,
                                             rf_source_signal_strength_dbm,
                                             rf_source_signal_frequency_mhz,
                                             terrain_x,
                                             terrain_y,
                                             terrain_z,
                                             geographic_coords,
                                             bin_dim_meters,
                                             max_ray_travel_meters,
                                             num_rays_per_source,
                                             min_receiver_signal_strength_dbm,
                                             assumed_source_height_above_ground,
                                             ray_step_bin_multiple,
                                             loop_grain_size,
                                             out_x,
                                             out_y,
                                             out_max_z,
                                             out_strongest_rf_source_id,
                                             out_max_rf_signal_strength_dbm);
}

template <typename S, typename T, typename Z>
int64_t rf_prop_impl(const S source_id,
                     const int64_t source_x_bin,
                     const int64_t source_y_bin,
                     const Z source_z,
                     const int64_t num_rays,
                     const T max_ray_travel_meters,
                     const T ray_step_in_bins,
                     const T assumed_source_height_above_ground,
                     const GeoRaster<T, Z>& geo_raster,
                     const RasterOffsets<T>& raster_offsets,
                     std::vector<T>& local_min_squared_distances,
                     std::vector<Z>& local_z) {
  const T numeric_min_sentinel{std::numeric_limits<T>::lowest()};
  const T numeric_max_sentinel{std::numeric_limits<T>::max()};

  const T start_x_bin = source_x_bin + 0.5;  // convert to fp up front
  const T start_y_bin = source_y_bin + 0.5;  // convert to fp up front
  const T angle_step = 2 * M_PI / num_rays;
  const T ray_step_meters = ray_step_in_bins * geo_raster.bin_dim_meters_;
  const int32_t num_ray_steps = max_ray_travel_meters / ray_step_meters;
  std::fill(local_min_squared_distances.begin(),
            local_min_squared_distances.end(),
            numeric_max_sentinel);

  int64_t num_bins_written{0};
  for (int32_t r = 0; r < num_rays; ++r) {
    const T theta_radians = r * angle_step;
    const T ray_x_step = cos(theta_radians) * ray_step_in_bins;
    const T ray_y_step = sin(theta_radians) * ray_step_in_bins;
    Z ray_step_z = source_z - assumed_source_height_above_ground;
    T max_obscured_z_slope = numeric_min_sentinel;
    // int32_t last_ray_step_raster_idx = bin_min_sentinel;
    for (int32_t s = 0; s < num_ray_steps; ++s) {
      const T ray_step_with_epsilon = s + ray_step_epsilon;
      const int32_t ray_step_x_bin =
          std::floor(start_x_bin + ray_step_with_epsilon * ray_x_step);
      const int32_t ray_step_y_bin =
          std::floor(start_y_bin + ray_step_with_epsilon * ray_y_step);
      if (geo_raster.is_bin_out_of_bounds(ray_step_x_bin, ray_step_y_bin)) {
        // Off grid, won't ever get back in so move to next ray
        break;
      }
      const int32_t ray_step_raster_idx =
          x_y_bin_to_bin_index(ray_step_x_bin, ray_step_y_bin, geo_raster.num_x_bins_);

      if (geo_raster.z_[ray_step_raster_idx] != geo_raster.null_sentinel_) {
        ray_step_z = geo_raster.z_[ray_step_raster_idx];
      }
      const T xy_meters_from_source = ray_step_with_epsilon * ray_step_meters;
      const T z_meters_from_source = ray_step_z - source_z;
      const T ray_step_z_slope = z_meters_from_source / xy_meters_from_source;
      if (ray_step_z_slope < max_obscured_z_slope) {
        continue;
      }
      max_obscured_z_slope = ray_step_z_slope;
      const T ray_step_distance_squared = xy_meters_from_source * xy_meters_from_source +
                                          z_meters_from_source * z_meters_from_source;
      const int32_t local_x_bin_idx = ray_step_x_bin - source_x_bin;
      const int32_t local_y_bin_idx = ray_step_y_bin - source_y_bin;
      const int32_t local_bin_idx =
          raster_offsets.getPermutedGridIdx(local_x_bin_idx, local_y_bin_idx);
      if (ray_step_distance_squared < local_min_squared_distances[local_bin_idx]) {
        if (local_min_squared_distances[local_bin_idx] == numeric_max_sentinel) {
          num_bins_written++;
        }
        local_min_squared_distances[local_bin_idx] = ray_step_distance_squared;
        local_z[local_bin_idx] = ray_step_z;
      }
    }
  }
  return num_bins_written;
}

template <typename S,
          typename T,
          typename Z,
          typename VecI,
          typename VecS,
          typename VecT,
          typename VecZ>
void write_rf_output(const S source_id,
                     const int64_t source_x_bin,
                     const int64_t source_y_bin,
                     const Z source_z,
                     const double source_signal_strength_dbm,
                     const double signal_frequency_mhz,
                     std::vector<T>& local_min_squared_distances,
                     std::vector<Z>& local_z,
                     const GeoRaster<T, Z>& geo_raster,
                     const RasterOffsets<T>& raster_offsets,
                     const int64_t start_output_slot,
                     VecI& out_grid_cell_id,
                     VecT& out_x,
                     VecT& out_y,
                     VecZ& out_max_z,
                     VecS& out_rf_source_id,
                     VecT& out_rf_signal_strength_dbm,
                     VecT& out_rf_signal_z_angle_degrees,
                     VecT& out_rf_source_distance_meters) {
  int64_t current_output_slot = start_output_slot;
  const int64_t per_source_max_bins = local_min_squared_distances.size();
  const T numeric_max_sentinel{std::numeric_limits<T>::max()};

  const T radians_to_degrees{180.0 / M_PI};

  for (int64_t b = 0; b < per_source_max_bins; ++b) {
    if (local_min_squared_distances[b] == numeric_max_sentinel) {
      continue;
    }
    const GridCell<T>& grid_cell = raster_offsets.getGridCell(b);
    const int32_t global_x_bin = source_x_bin + grid_cell.x_bin_idx;
    const int32_t global_y_bin = source_y_bin + grid_cell.y_bin_idx;

    out_grid_cell_id[current_output_slot] =
        x_y_bin_to_bin_index(global_x_bin, global_y_bin, geo_raster.num_x_bins_);
    out_x[current_output_slot] =
        (global_x_bin + 0.5) * geo_raster.x_scale_bin_to_input_ + geo_raster.x_min_;
    out_y[current_output_slot] =
        (global_y_bin + 0.5) * geo_raster.y_scale_bin_to_input_ + geo_raster.y_min_;
    out_max_z[current_output_slot] = local_z[b];
    out_rf_source_id[current_output_slot] = source_id;
    const T rf_source_distance_meters = sqrt(local_min_squared_distances[b]);
    // const T rf_source_distance_meters = grid_cell.xy_distance;
    out_rf_signal_strength_dbm[current_output_slot] =
        source_signal_strength_dbm - (20.0 * log10(rf_source_distance_meters) +
                                      20.0 * log10(signal_frequency_mhz) - 27.55);
    out_rf_signal_z_angle_degrees[current_output_slot] =
        asin((source_z - local_z[b]) / rf_source_distance_meters) * radians_to_degrees;
    // out_rf_signal_z_angle_degrees[current_output_slot] =
    //    atan2(source_z - local_z[b], rf_source_distance_meters) * radians_to_degrees;
    out_rf_source_distance_meters[current_output_slot] = rf_source_distance_meters;

    ++current_output_slot;
  }
}

template <typename VecTypeI, typename VecTypeS, typename VecTypeT>
std::vector<int64_t> filter_top_k_sources_by_grid_cell(
    const VecTypeI& out_terrain_bin_id,
    const VecTypeS& out_source_id,
    const VecTypeT& out_signal_strength,
    const size_t num_valid_entries,
    const int64_t top_k_sources_per_terrain_bin) {
  auto timer = DEBUG_TIMER(__func__);
  if (num_valid_entries == 0) {
    std::vector<int64_t> return_vec;
    return return_vec;
  }
  // Todo: error if we have more than 2^31 input rows until we can support larger result
  // sets from UDTFs
  const size_t tbb_loop_grain_size = static_cast<int64_t>(10000);

  std::vector<int32_t> permutation_idxs(num_valid_entries);

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, num_valid_entries, tbb_loop_grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        const int64_t r_end = r.end();
        for (int64_t p = r.begin(); p < r_end; ++p) {
          permutation_idxs[p] = p;
        }
      });

  tbb::parallel_sort(permutation_idxs.begin(),
                     permutation_idxs.begin() + num_valid_entries,
                     [&](const int64_t& a, const int64_t& b) {
                       if (out_terrain_bin_id[a] < out_terrain_bin_id[b]) {
                         return true;
                       } else if (out_terrain_bin_id[a] > out_terrain_bin_id[b]) {
                         return false;
                       }
                       return out_signal_strength[a] > out_signal_strength[b];
                     });

  const int32_t grid_cell_min = out_terrain_bin_id[permutation_idxs[0]];
  const int32_t grid_cell_max =
      out_terrain_bin_id[permutation_idxs[num_valid_entries - 1]];
  const int32_t grid_cell_range = grid_cell_max - grid_cell_min + 1;

  std::vector<int64_t> grid_cell_counts(grid_cell_range, 0);
  const int64_t max_thread_range = 2 * ((num_valid_entries / tbb_loop_grain_size) + 1);
  std::vector<int64_t> unwritten_grid_cells(max_thread_range, -1);
  std::vector<int64_t> unwritten_counts(max_thread_range);
  std::atomic<int64_t> thread_range_counter = 0;
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, num_valid_entries, tbb_loop_grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        const int64_t r_begin = r.begin();
        const int64_t r_end = r.end();
        // Skip all cells at the beginning of our range with the first value so we don't
        // have to worry about contending with the thread with the range before us
        const int64_t this_thread_range_id =
            thread_range_counter.fetch_add(1, std::memory_order_relaxed);
        const int64_t non_offset_start_grid_cell =
            out_terrain_bin_id[permutation_idxs[r_begin]];
        unwritten_grid_cells[this_thread_range_id] =
            non_offset_start_grid_cell - grid_cell_min;
        int64_t p = r_begin + 1;
        while (p < r_end &&
               out_terrain_bin_id[permutation_idxs[p]] == non_offset_start_grid_cell) {
          p++;
        }
        unwritten_counts[this_thread_range_id] = p - r_begin;
        for (; p < r_end; ++p) {
          grid_cell_counts[out_terrain_bin_id[permutation_idxs[p]] - grid_cell_min]++;
        }
      },
      tbb::simple_partitioner());
  // Now write out the first values we passed over single threaded
  for (int64_t t = 0; t < max_thread_range; ++t) {
    const int64_t unwritten_grid_cell = unwritten_grid_cells[t];
    if (unwritten_grid_cell >= 0) {
      grid_cell_counts[unwritten_grid_cell] += unwritten_counts[t];
    }
  }

  std::vector<int64_t> input_prefix_sum(grid_cell_range + 1);
  std::vector<int64_t> output_prefix_sum(grid_cell_range + 1);
  input_prefix_sum[0] = 0;
  output_prefix_sum[0] = 0;
  for (int64_t grid_cell_idx = 0; grid_cell_idx < grid_cell_range; ++grid_cell_idx) {
    const int64_t grid_cell_count = grid_cell_counts[grid_cell_idx];
    input_prefix_sum[grid_cell_idx + 1] =
        input_prefix_sum[grid_cell_idx] + grid_cell_count;
    output_prefix_sum[grid_cell_idx + 1] =
        output_prefix_sum[grid_cell_idx] +
        std::min(grid_cell_count, top_k_sources_per_terrain_bin);
  }

  const size_t num_output_rows = output_prefix_sum[grid_cell_range];
  std::vector<int64_t> output_row_idxs(num_output_rows);

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, grid_cell_range, tbb_loop_grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        const int64_t r_end = r.end();
        for (int64_t i = r.begin(); i < r_end; ++i) {
          const int64_t input_start_idx = input_prefix_sum[i];
          const int64_t output_start_idx = output_prefix_sum[i];
          const int64_t num_output_entries = output_prefix_sum[i + 1] - output_start_idx;
          for (int64_t r = 0; r < num_output_entries; ++r) {
            output_row_idxs[output_start_idx + r] = permutation_idxs[input_start_idx + r];
          }
        }
      });

  return output_row_idxs;
}

template <typename T>
void project_filtered_column(Column<T>& data_col,
                             const std::vector<int64_t>& output_row_idxs) {
  const int64_t num_output_rows = output_row_idxs.size();
  std::vector<T> temp_output_buffer(num_output_rows);
  const size_t tbb_loop_grain_size = static_cast<int64_t>(1000);
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_output_rows, tbb_loop_grain_size),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const int64_t r_end = r.end();
                      for (int64_t out_idx = r.begin(); out_idx < r_end; ++out_idx) {
                        temp_output_buffer[out_idx] = data_col[output_row_idxs[out_idx]];
                      }
                    });

  std::copy(temp_output_buffer.begin(), temp_output_buffer.end(), data_col.ptr_);
}

// clang-format off
/*
  UDTF: tf_rf_prop__cpu_template(TableFunctionManager,
  Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> repeater_height_meters> rf_sources,
  bool rf_source_z_is_relative_to_terrain | default = true, double rf_source_signal_strength_dbm,
  double rf_source_signal_frequency_mhz | default = 3900.0,
  Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations,
  bool geographic_coords | default = true,
  double bin_dim_meters,
  int64_t strongest_k_sources_per_terrain_bin,
  double max_ray_travel_meters | default = 3000.0,
  int64_t num_rays_per_source | default = 1440,
  double min_receiver_signal_strength_dbm | default = -120.0,
  double default_source_height_agl_meters | default = 20.0,
  double ray_step_bin_multiple | default = 1.0,
  int64_t loop_grain_size | default=64) | filter_table_function_transpose=on ->
  Column<int32_t> terrain_bin_id, Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters,
  Column<S> rf_source_id | input_id=args<0>, Column<T1> rf_signal_strength_dbm,
  Column<T1> rf_signal_z_angle_degrees, Column<T1> rf_source_distance_meters,
  S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2>
TEMPLATE_NOINLINE int32_t
tf_rf_prop__cpu_template(TableFunctionManager& mgr,
                         const Column<S>& rf_source_id,
                         const Column<T1>& rf_source_x,
                         const Column<T1>& rf_source_y,
                         const Column<Z1>& rf_source_z,
                         const bool rf_source_z_is_relative_to_terrain,
                         const double rf_source_signal_strength_dbm,
                         const double rf_source_signal_frequency_mhz,
                         const Column<T2>& terrain_x,
                         const Column<T2>& terrain_y,
                         const Column<Z2>& terrain_z,
                         const bool geographic_coords,
                         const double bin_dim_meters,
                         const int64_t num_top_sources_per_terrain_bin,
                         const double max_ray_travel_meters,
                         const int64_t num_rays_per_source,
                         const double min_receiver_signal_strength_dbm,
                         const double assumed_source_height_above_ground,
                         const double ray_step_bin_multiple,
                         const int64_t loop_grain_size,
                         Column<int32_t>& out_terrain_bin_id,
                         Column<T1>& out_x,
                         Column<T1>& out_y,
                         Column<Z1>& out_max_z,
                         Column<S>& out_rf_source_id,
                         Column<T1>& out_rf_signal_strength_dbm,
                         Column<T1>& out_rf_signal_z_angle_degrees,
                         Column<T1>& out_rf_source_distance_meters) {
  auto timer = DEBUG_TIMER(__func__);

  if (!is_valid_tf_input(rf_source_signal_frequency_mhz,
                         0.0,
                         BoundsType::Min,
                         IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("rf_source_signal_frequency_mhz must be > 0");
  }
  if (!is_valid_tf_input(bin_dim_meters, 0.0, BoundsType::Min, IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("bin_dim_meters must be > 0");
  }

  if (!is_valid_tf_input(
          max_ray_travel_meters, 0.0, BoundsType::Min, IntervalType::Exclusive)) {
    return mgr.ERROR_MESSAGE("max_ray_travel_meters must be > 0");
  }
  if (!is_valid_tf_input(assumed_source_height_above_ground,
                         0.0,
                         BoundsType::Min,
                         IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("assumed_source_height_above_ground must be >= 0");
  }
  if (!is_valid_tf_input(loop_grain_size,
                         static_cast<decltype(loop_grain_size)>(1),
                         BoundsType::Min,
                         IntervalType::Inclusive)) {
    return mgr.ERROR_MESSAGE("loop_grain_size must be >= 1");
  }

  // Note: The following input values are unchecked:
  // rf_source_signal_strength_dbm and min_receiver_signal_strength_dbm have unbounded
  // ranges num_top_sources_per_terrain_bin will fetch all results if <= 0
  // num_rays_per_source and ray_step_bin_multiple set automatic defaults if values are <=
  // 0

  GeoRaster<T1, Z1> geo_raster(terrain_x,
                               terrain_y,
                               terrain_z,
                               RasterAggType::MAX,
                               bin_dim_meters,
                               geographic_coords,
                               true);

  if (geo_raster.num_bins_ == 0) {
    return mgr.ERROR_MESSAGE("terrain bins are empty");
  }
  geo_raster.fill_bins_from_neighbors(1 /* fill_radius */, true /* fill_only_nulls*/);
  const double max_ray_travel_meters_for_min_signal =
      get_min_distance_for_power_frequency(min_receiver_signal_strength_dbm,
                                           rf_source_signal_strength_dbm,
                                           rf_source_signal_frequency_mhz);
  const double max_ray_travel_meters_calculated =
      std::min(max_ray_travel_meters, max_ray_travel_meters_for_min_signal);

  const RasterOffsets<T1> raster_offsets(max_ray_travel_meters_calculated,
                                         bin_dim_meters,
                                         false /* -> only_first_quarter_radian */);
  const T1 ray_step_in_bins =
      (ray_step_bin_multiple > 0.0 &&
               ray_step_bin_multiple < max_ray_travel_meters_calculated
           ? ray_step_bin_multiple
           : 1.0);
  const int32_t num_rays = num_rays_per_source > 0 && num_rays_per_source < 10000
                               ? num_rays_per_source
                               : raster_offsets.num_edge_bins_ * 8;
  const int64_t per_source_max_bins = raster_offsets.num_valid_grid_bins_;
  const size_t num_rf_sources = rf_source_id.size();

  const int32_t partition_size = max_ray_travel_meters_calculated / bin_dim_meters;
  const int32_t num_x_partitions =
      geo_raster.num_x_bins_ / partition_size +
      static_cast<int32_t>((geo_raster.num_x_bins_ % partition_size) > 0);
  const int32_t num_y_partitions =
      geo_raster.num_y_bins_ / partition_size +
      static_cast<int32_t>((geo_raster.num_y_bins_ % partition_size) > 0);
  // const int32_t num_partitions = num_x_partitions * num_y_partitions;
  // const int32_t num_local_bins_per_dim = partition_size * 3;
  // const int32_t num_local_bins = num_local_bins_per_dim * num_local_bins_per_dim;

  const PartitionInfo partition_info =
      get_partition_info<T1>(rf_source_x,
                             rf_source_y,
                             geo_raster.x_min_,
                             geo_raster.x_scale_input_to_bin_,
                             geo_raster.y_min_,
                             geo_raster.y_scale_input_to_bin_,
                             partition_size,
                             num_x_partitions,
                             num_y_partitions);

  const int64_t max_output_size = num_rf_sources * per_source_max_bins;
  if (out_x.size() == 0) {
    // allow function to work with pre-allocated output buffers
    mgr.set_output_row_size(max_output_size);
  }
  std::atomic<int64_t> output_row_count = 0;

  const size_t tbb_loop_grain_size = static_cast<int64_t>(loop_grain_size);
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_rf_sources, tbb_loop_grain_size),
      [&](const tbb::blocked_range<size_t>& r) {
        std::vector<T1> local_min_squared_distances(per_source_max_bins);
        std::vector<Z1> local_z(per_source_max_bins);
        for (size_t source_idx = r.begin(); source_idx != r.end(); ++source_idx) {
          const S source_id = rf_source_id[source_idx];
          const int64_t source_x_bin = (rf_source_x[source_idx] - geo_raster.x_min_) *
                                       geo_raster.x_scale_input_to_bin_;
          const int64_t source_y_bin = (rf_source_y[source_idx] - geo_raster.y_min_) *
                                       geo_raster.y_scale_input_to_bin_;
          if (geo_raster.is_bin_out_of_bounds(source_x_bin, source_y_bin)) {
            continue;
          }
          const Z1 source_z_input = rf_source_z[source_idx];
          const Z1 source_z = rf_source_z_is_relative_to_terrain
                                  ? geo_raster.offset_source_z_from_raster_z(
                                        source_x_bin, source_y_bin, source_z_input)
                                  : source_z_input;
          if (geo_raster.is_null(source_z)) {
            continue;
          }
          const int64_t num_bins_written =
              rf_prop_impl<S, T1, Z1>(source_id,
                                      source_x_bin,
                                      source_y_bin,
                                      source_z,
                                      num_rays,
                                      max_ray_travel_meters_calculated,
                                      ray_step_in_bins,
                                      assumed_source_height_above_ground,
                                      geo_raster,
                                      raster_offsets,
                                      local_min_squared_distances,
                                      local_z);

          if (num_bins_written > 0) {
            const int64_t start_output_slot =
                output_row_count.fetch_add(num_bins_written, std::memory_order_relaxed);
            write_rf_output(source_id,
                            source_x_bin,
                            source_y_bin,
                            source_z,
                            rf_source_signal_strength_dbm,
                            rf_source_signal_frequency_mhz,
                            local_min_squared_distances,
                            local_z,
                            geo_raster,
                            raster_offsets,
                            start_output_slot,
                            out_terrain_bin_id,
                            out_x,
                            out_y,
                            out_max_z,
                            out_rf_source_id,
                            out_rf_signal_strength_dbm,
                            out_rf_signal_z_angle_degrees,
                            out_rf_source_distance_meters);
          }
        }
      });

  if (num_top_sources_per_terrain_bin >= 1 &&
      num_top_sources_per_terrain_bin < static_cast<int64_t>(num_rf_sources)) {
    const std::vector<int64_t> filtered_row_indexes =
        filter_top_k_sources_by_grid_cell(out_terrain_bin_id,
                                          out_rf_source_id,
                                          out_rf_signal_strength_dbm,
                                          output_row_count,
                                          num_top_sources_per_terrain_bin);
    project_filtered_column(out_terrain_bin_id, filtered_row_indexes);
    project_filtered_column(out_x, filtered_row_indexes);
    project_filtered_column(out_y, filtered_row_indexes);
    project_filtered_column(out_max_z, filtered_row_indexes);
    project_filtered_column(out_rf_source_id, filtered_row_indexes);
    project_filtered_column(out_rf_signal_strength_dbm, filtered_row_indexes);
    project_filtered_column(out_rf_signal_z_angle_degrees, filtered_row_indexes);
    project_filtered_column(out_rf_source_distance_meters, filtered_row_indexes);
    return filtered_row_indexes.size();
  }
  return output_row_count;
}

// clang-format off
/*
  UDTF: tf_rf_prop__cpu_template(TableFunctionManager,
  Cursor<Column<S> rf_source_id, Column<T1> x, Column<T1> y, Column<Z1> repeater_height_meters> rf_sources,
  bool rf_source_z_is_relative_to_terrain | default = true, double rf_source_signal_strength_dbm,
  double rf_source_signal_frequency_mhz,
  Cursor<Column<T2> x, Column<T2> y, Column<Z2> elevation_amsl_meters> terrain_elevations,
  double bin_dim_meters,
  int64_t num_top_sources_per_terrain_bin,
  double max_ray_travel_meters | default = 3000.0, 
  int64_t num_rays_per_source | default = 1440,
  double min_receiver_signal_strength_dbm | default = -120.0) | filter_table_function_transpose=on ->
  Column<int32_t> terrain_bin_id, Column<T1> x, Column<T1> y, Column<Z1> elevation_amsl_meters,
  Column<S> rf_source_id | input_id=args<0>, Column<T1> rf_signal_strength_dbm,
  Column<T1> rf_signal_z_angle_degrees, Column<T1> rf_source_distance_meters,
  S=[int64_t, TextEncodingDict], T1=[double], Z1=[double], T2=[double], Z2=[double]
 */
// clang-format on

template <typename S, typename T1, typename Z1, typename T2, typename Z2>
TEMPLATE_NOINLINE int32_t
tf_rf_prop__cpu_template(TableFunctionManager& mgr,
                         const Column<S>& rf_source_id,
                         const Column<T1>& rf_source_x,
                         const Column<T1>& rf_source_y,
                         const Column<Z1>& rf_source_z,
                         const bool rf_source_z_is_relative_to_terrain,
                         const double rf_source_signal_strength_dbm,
                         const double rf_source_signal_frequency_mhz,
                         const Column<T2>& terrain_x,
                         const Column<T2>& terrain_y,
                         const Column<Z2>& terrain_z,
                         const double bin_dim_meters,
                         const int64_t num_top_sources_per_terrain_bin,
                         const double max_ray_travel_meters,
                         const int64_t num_rays_per_source,
                         const double min_receiver_signal_strength_dbm,
                         Column<int32_t>& out_grid_cell_id,
                         Column<T1>& out_x,
                         Column<T1>& out_y,
                         Column<Z1>& out_max_z,
                         Column<S>& out_rf_source_id,
                         Column<T1>& out_rf_signal_strength_dbm,
                         Column<T1>& out_rf_signal_z_angle_degrees,
                         Column<T1>& out_rf_source_distance_meters) {
  return tf_rf_prop__cpu_template(mgr,
                                  rf_source_id,
                                  rf_source_x,
                                  rf_source_y,
                                  rf_source_z,
                                  rf_source_z_is_relative_to_terrain,
                                  rf_source_signal_strength_dbm,
                                  rf_source_signal_frequency_mhz,
                                  terrain_x,
                                  terrain_y,
                                  terrain_z,
                                  true,
                                  bin_dim_meters,
                                  num_top_sources_per_terrain_bin,
                                  max_ray_travel_meters,
                                  num_rays_per_source,
                                  min_receiver_signal_strength_dbm,
                                  10.0,
                                  1.0,
                                  40,
                                  out_grid_cell_id,
                                  out_x,
                                  out_y,
                                  out_max_z,
                                  out_rf_source_id,
                                  out_rf_signal_strength_dbm,
                                  out_rf_signal_z_angle_degrees,
                                  out_rf_source_distance_meters);
}

#endif  // __CUDACC__
#endif  // HAVE_RF_PROP_TFS
