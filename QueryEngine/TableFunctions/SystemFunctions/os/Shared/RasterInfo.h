/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__

#include <cstdint>

namespace RasterFormat_Namespace {

struct RasterInfo {
  const int64_t raster_channels;
  const int64_t x_pixels_per_tile;
  const int64_t y_pixels_per_tile;
  const int64_t halo_x_pixels_per_tile_boundary;
  const int64_t halo_y_pixels_per_tile_boundary;
  const int64_t logical_x_pixels_per_tile;
  const int64_t logical_y_pixels_per_tile;
  const int64_t x_tiles;
  const int64_t y_tiles;
  const int64_t batch_tiles;
  const double x_input_units_per_pixel;
  const double y_input_units_per_pixel;
  const double min_x_input;
  const double min_y_input;
};

}  // namespace RasterFormat_Namespace

#endif  // #ifdef __CUDACC__