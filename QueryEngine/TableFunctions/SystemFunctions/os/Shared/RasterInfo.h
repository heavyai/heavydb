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