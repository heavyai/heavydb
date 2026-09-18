/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace omniverse_connector {

enum class MeshType { kInterpolated = 0, kStepped = 1 };

template <typename TXY, typename TA>
int8_t* export_terrain_texture(const int32_t grid_size_x,
                               const int32_t grid_size_y,
                               const TXY x_min,
                               const TXY y_min,
                               const TXY x_scale_input_to_bin,
                               const TXY y_scale_input_to_bin,
                               const TXY* x,
                               const TXY* y,
                               const TA** attrs,
                               const int32_t num_attrs,
                               const int64_t num_rows,
                               int32_t& num_png_bytes);

template <typename TA>
int8_t* export_buildings_texture(const int64_t* rowid,
                                 const TA** attrs,
                                 const int32_t num_attrs,
                                 const int64_t num_rows,
                                 int32_t& num_patches_xy,
                                 int32_t& num_png_bytes);

template <typename TXY, typename TZ>
int32_t* export_grid_mesh(const MeshType mesh_type,
                          const double tile_origin_x,
                          const double tile_origin_y,
                          const int32_t grid_size_x,
                          const int32_t grid_size_y,
                          const TXY x_min,
                          const TXY y_min,
                          const TXY x_scale_input_to_bin,
                          const TXY y_scale_input_to_bin,
                          const TXY* x,
                          const TXY* y,
                          const TZ* z,
                          const int64_t num_rows,
                          int32_t& mesh_data_size);

void export_free(void* p);

void export_polygons(const int64_t num_rows,
                     const int64_t* rowids,
                     const std::vector<std::vector<double>>& row_coords,
                     const std::vector<std::vector<int32_t>>& row_ring_sizes,
                     const std::vector<std::vector<int32_t>>& row_poly_rings,
                     const double* row_centroids_x,
                     const double* row_centroids_y,
                     const float* row_base_elevations,
                     const std::vector<std::vector<float>>& row_heights,
                     int32_t**& row_mesh_data,
                     int32_t*& row_mesh_data_size);

}  // namespace omniverse_connector
