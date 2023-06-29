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

#include <tbb/parallel_for.h>
#include "CpuTimer.hpp"
#include "QueryEngine/heavydbTypes.h"
#include "RasterInfo.h"
#include "TableFunctionsCommon.hpp"

namespace RasterFormat_Namespace {

template <typename OutputType>
TEMPLATE_NOINLINE void fill_row_halo(std::vector<OutputType>& output_data,
                                     const int64_t tile_x,
                                     const int64_t tile_y,
                                     const int64_t tiles_x,
                                     const int64_t tiles_y,
                                     const int64_t halo_y_pixels_per_tile_boundary,
                                     const int64_t x_pixels_per_tile,
                                     const int64_t y_pixels_per_tile,
                                     const int64_t num_raster_channels,
                                     const int64_t source_tile_y_offset) {
  if (tile_y + source_tile_y_offset >= tiles_y || tile_y + source_tile_y_offset < 0) {
    return;
  }
  const int64_t output_tile_slot = tile_y * tiles_x + tile_x;
  const int64_t source_tile_slot = (tile_y + source_tile_y_offset) * tiles_x + tile_x;
  const int64_t pixels_per_tile = x_pixels_per_tile * y_pixels_per_tile;
  const int64_t channel_pixels_per_tile = pixels_per_tile * num_raster_channels;
  int64_t halo_y_start;
  int64_t source_y_start;
  int64_t halo_y_increment;
  int64_t source_y_increment;
  if (source_tile_y_offset > 0) {
    // fill in top/north halo
    halo_y_start = y_pixels_per_tile - 1;
    halo_y_increment = -1;
    source_y_start = 2 * halo_y_pixels_per_tile_boundary - 1;
    source_y_increment = -1;
  } else {
    // fill in bottom/south halo
    halo_y_start = 0;
    halo_y_increment = 1;
    source_y_start = y_pixels_per_tile - 2 * halo_y_pixels_per_tile_boundary + 1;
    source_y_increment = 1;
  }

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, halo_y_pixels_per_tile_boundary),
      [&](const tbb::blocked_range<int64_t>& halo_y_pixels_range) {
        const int64_t start_halo_y_offset = halo_y_pixels_range.begin();
        const int64_t end_halo_y_offset = halo_y_pixels_range.end();
        for (int64_t halo_y_offset = start_halo_y_offset;
             halo_y_offset < end_halo_y_offset;
             ++halo_y_offset) {
          const int64_t halo_y = halo_y_start + halo_y_offset * halo_y_increment;
          const int64_t source_y = source_y_start + halo_y_offset * source_y_increment;
          const int64_t output_starting_slot_for_halo_row =
              output_tile_slot * channel_pixels_per_tile + halo_y * x_pixels_per_tile;
          const int64_t source_starting_slot_for_halo_row =
              source_tile_slot * channel_pixels_per_tile + source_y * x_pixels_per_tile;
          for (int64_t local_x_pixel = 0; local_x_pixel < x_pixels_per_tile;
               ++local_x_pixel) {
            const int64_t output_starting_slot =
                output_starting_slot_for_halo_row + local_x_pixel;
            const int64_t source_starting_slot =
                source_starting_slot_for_halo_row + local_x_pixel;
            output_data[output_starting_slot] = output_data[source_starting_slot];
            output_data[output_starting_slot + pixels_per_tile] =
                output_data[source_starting_slot + pixels_per_tile];
            output_data[output_starting_slot + 2 * pixels_per_tile] =
                output_data[source_starting_slot + 2 * pixels_per_tile];
          }
        }
      });
}

template <typename OutputType>
TEMPLATE_NOINLINE void fill_col_halo(std::vector<OutputType>& output_data,
                                     const int64_t tile_x,
                                     const int64_t tile_y,
                                     const int64_t tiles_x,
                                     const int64_t tiles_y,
                                     const int64_t halo_x_pixels_per_tile_boundary,
                                     const int64_t x_pixels_per_tile,
                                     const int64_t y_pixels_per_tile,
                                     const int64_t num_raster_channels,
                                     const int64_t source_tile_x_offset) {
  if (tile_x + source_tile_x_offset >= tiles_x || tile_x + source_tile_x_offset < 0) {
    return;
  }
  const int64_t output_tile_slot = tile_y * tiles_x + tile_x;
  const int64_t source_tile_slot = tile_y * tiles_x + tile_x + source_tile_x_offset;
  const int64_t pixels_per_tile = x_pixels_per_tile * y_pixels_per_tile;
  const int64_t channel_pixels_per_tile = pixels_per_tile * num_raster_channels;
  int64_t halo_x_start;
  int64_t source_x_start;
  int64_t halo_x_increment;
  int64_t source_x_increment;
  if (source_tile_x_offset > 0) {
    // fill in right/east halo
    halo_x_start = x_pixels_per_tile - 1;
    halo_x_increment = -1;
    source_x_start = 2 * halo_x_pixels_per_tile_boundary - 1;
    source_x_increment = -1;
  } else {
    // fill in left/west halo
    halo_x_start = 0;
    halo_x_increment = 1;
    source_x_start = x_pixels_per_tile - 2 * halo_x_pixels_per_tile_boundary + 1;
    source_x_increment = 1;
  }

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, y_pixels_per_tile),
      [&](const tbb::blocked_range<int64_t>& y_pixels_range) {
        const int64_t start_local_y_pixel = y_pixels_range.begin();
        const int64_t end_local_y_pixel = y_pixels_range.end();

        for (int64_t local_y_pixel = start_local_y_pixel;
             local_y_pixel < end_local_y_pixel;
             ++local_y_pixel) {
          const int64_t output_starting_slot_for_halo_row =
              output_tile_slot * channel_pixels_per_tile +
              local_y_pixel * x_pixels_per_tile;
          const int64_t source_starting_slot_for_halo_row =
              source_tile_slot * channel_pixels_per_tile +
              local_y_pixel * x_pixels_per_tile;

          for (int64_t halo_x_offset = 0; halo_x_offset < halo_x_pixels_per_tile_boundary;
               ++halo_x_offset) {
            const int64_t halo_x = halo_x_start + halo_x_offset * halo_x_increment;
            const int64_t source_x = source_x_start + halo_x_offset * source_x_increment;
            const int64_t output_starting_slot =
                output_starting_slot_for_halo_row + halo_x;
            const int64_t source_starting_slot =
                source_starting_slot_for_halo_row + source_x;
            output_data[output_starting_slot] = output_data[source_starting_slot];
            output_data[output_starting_slot + pixels_per_tile] =
                output_data[source_starting_slot + pixels_per_tile];
            output_data[output_starting_slot + 2 * pixels_per_tile] =
                output_data[source_starting_slot + 2 * pixels_per_tile];
          }
        }
      });
}

template <typename PixelType, typename ColorType, typename OutputType>
TEMPLATE_NOINLINE std::pair<std::vector<OutputType>, RasterInfo> format_raster_data(
    const Column<PixelType>& raster_x,
    const Column<PixelType>& raster_y,
    const ColumnList<ColorType>& raster_channels,
    const PixelType x_input_units_per_pixel,
    const PixelType y_input_units_per_pixel,
    const float max_color_val,
    const int64_t x_pixels_per_tile,
    const int64_t y_pixels_per_tile,
    const int64_t tile_boundary_halo_pixels,
    const int64_t target_batch_size_multiple,
    std::shared_ptr<CpuTimer> timer) {
  const double x_pixels_per_input_unit = 1.0 / x_input_units_per_pixel;
  const double y_pixels_per_input_unit = 1.0 / y_input_units_per_pixel;
  timer->start_event_timer("Compute raster metadata");
  const auto x_pixel_range = get_column_min_max(raster_x);
  const auto y_pixel_range = get_column_min_max(raster_y);

  const int64_t logical_x_pixels =
      (x_pixel_range.second - x_pixel_range.first) * x_pixels_per_input_unit + 1;
  const auto min_x_input_unit = x_pixel_range.first;

  const int64_t logical_y_pixels =
      (y_pixel_range.second - y_pixel_range.first) * y_pixels_per_input_unit + 1;
  const auto min_y_input_unit = y_pixel_range.first;

  int64_t x_tiles = 1;
  int64_t y_tiles = 1;
  int64_t halo_x_pixels_per_tile_boundary = 0;
  int64_t halo_y_pixels_per_tile_boundary = 0;
  int64_t logical_x_pixels_per_tile = x_pixels_per_tile;
  int64_t logical_y_pixels_per_tile = y_pixels_per_tile;
  if (logical_x_pixels > x_pixels_per_tile || logical_y_pixels > y_pixels_per_tile) {
    // We need to consider padding if we cannot fit in single buffer
    halo_x_pixels_per_tile_boundary =
        std::min(tile_boundary_halo_pixels, (x_pixels_per_tile - 1) / 2);
    halo_y_pixels_per_tile_boundary =
        std::min(tile_boundary_halo_pixels, (y_pixels_per_tile - 1) / 2);
    logical_x_pixels_per_tile -= halo_x_pixels_per_tile_boundary * 2;
    logical_y_pixels_per_tile -= halo_y_pixels_per_tile_boundary * 2;
    x_tiles =
        (logical_x_pixels + (logical_x_pixels_per_tile - 1)) / logical_x_pixels_per_tile;
    y_tiles =
        (logical_y_pixels + (logical_y_pixels_per_tile - 1)) / logical_y_pixels_per_tile;
  }

  const int64_t num_input_pixels = raster_x.size();
  const int64_t num_raster_channels = raster_channels.numCols();

  const float normalize_ratio = 1.0f / max_color_val;
  const int64_t num_tiles = x_tiles * y_tiles;
  const int64_t batch_tiles =
      num_tiles == 1 ? 1
                     : (num_tiles + target_batch_size_multiple - 1) /
                           target_batch_size_multiple * target_batch_size_multiple;
  const int64_t batch_pixels = batch_tiles * x_pixels_per_tile * y_pixels_per_tile;
  std::vector<OutputType> output_data(batch_pixels * num_raster_channels,
                                      static_cast<OutputType>(0));
  const RasterInfo raster_info{num_raster_channels,
                               x_pixels_per_tile,
                               y_pixels_per_tile,
                               halo_x_pixels_per_tile_boundary,
                               halo_y_pixels_per_tile_boundary,
                               logical_x_pixels_per_tile,
                               logical_y_pixels_per_tile,
                               x_tiles,
                               y_tiles,
                               batch_tiles,
                               static_cast<double>(x_input_units_per_pixel),
                               static_cast<double>(y_input_units_per_pixel),
                               static_cast<double>(min_x_input_unit),
                               static_cast<double>(min_y_input_unit)};

  timer->start_event_timer("Format logical pixels");
  const int64_t pixels_per_tile = x_pixels_per_tile * y_pixels_per_tile;
  if (x_tiles == 1 && y_tiles == 1) {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, num_input_pixels),
        [&](const tbb::blocked_range<int64_t>& input_pixel_range) {
          const int64_t end_input_pixel_idx = input_pixel_range.end();
          for (int64_t input_pixel_idx = input_pixel_range.begin();
               input_pixel_idx < end_input_pixel_idx;
               ++input_pixel_idx) {
            const int64_t output_pixel_x = round(
                (raster_x[input_pixel_idx] - min_x_input_unit) * x_pixels_per_input_unit);
            const int64_t output_pixel_y = round(
                (raster_y[input_pixel_idx] - min_y_input_unit) * y_pixels_per_input_unit);
            const int64_t output_starting_slot =
                output_pixel_y * x_pixels_per_tile + output_pixel_x;
            for (int64_t channel_idx = 0; channel_idx < num_raster_channels;
                 ++channel_idx) {
              output_data[output_starting_slot + channel_idx * pixels_per_tile] =
                  std::min(
                      static_cast<float>(raster_channels[channel_idx][input_pixel_idx]) *
                          normalize_ratio,
                      1.0f);
            }
          }
        });
  } else {
    const int64_t channel_pixels_per_tile = pixels_per_tile * num_raster_channels;
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, num_input_pixels),
        [&](const tbb::blocked_range<int64_t>& input_pixel_range) {
          const int64_t end_input_pixel_idx = input_pixel_range.end();
          for (int64_t input_pixel_idx = input_pixel_range.begin();
               input_pixel_idx < end_input_pixel_idx;
               ++input_pixel_idx) {
            const int64_t logical_output_pixel_x = round(
                (raster_x[input_pixel_idx] - min_x_input_unit) * x_pixels_per_input_unit);
            const int64_t logical_output_pixel_y = round(
                (raster_y[input_pixel_idx] - min_y_input_unit) * y_pixels_per_input_unit);
            const int64_t output_tile_x =
                logical_output_pixel_x / logical_x_pixels_per_tile;
            const int64_t output_tile_y =
                logical_output_pixel_y / logical_y_pixels_per_tile;
            const int64_t local_output_pixel_x =
                (logical_output_pixel_x % logical_x_pixels_per_tile) +
                halo_x_pixels_per_tile_boundary;
            const int64_t local_output_pixel_y =
                (logical_output_pixel_y % logical_y_pixels_per_tile) +
                halo_y_pixels_per_tile_boundary;
            const int64_t output_tile_slot = output_tile_y * x_tiles + output_tile_x;
            const int64_t output_starting_slot =
                output_tile_slot * channel_pixels_per_tile +
                local_output_pixel_y * x_pixels_per_tile + local_output_pixel_x;
            for (int64_t channel_idx = 0; channel_idx < num_raster_channels;
                 ++channel_idx) {
              output_data[output_starting_slot + channel_idx * pixels_per_tile] =
                  std::min(
                      static_cast<float>(raster_channels[channel_idx][input_pixel_idx]) *
                          normalize_ratio,
                      1.0f);
            }
          }
        });

    // Now fill in halos
    if (halo_x_pixels_per_tile_boundary > 0 || halo_y_pixels_per_tile_boundary > 0) {
      timer->start_event_timer("Add halos");
      for (int64_t tile_y = 0; tile_y < y_tiles; ++tile_y) {
        for (int64_t tile_x = 0; tile_x < x_tiles; ++tile_x) {
          for (int64_t offset = -1; offset < 1; offset += 2) {
            fill_row_halo(output_data,
                          tile_x,
                          tile_y,
                          x_tiles,
                          y_tiles,
                          halo_y_pixels_per_tile_boundary,
                          x_pixels_per_tile,
                          y_pixels_per_tile,
                          num_raster_channels,
                          offset);
          }
        }
      }
      for (int64_t tile_y = 0; tile_y < y_tiles; ++tile_y) {
        for (int64_t tile_x = 0; tile_x < x_tiles; ++tile_x) {
          fill_col_halo(output_data,
                        tile_x,
                        tile_y,
                        x_tiles,
                        y_tiles,
                        halo_x_pixels_per_tile_boundary,
                        x_pixels_per_tile,
                        y_pixels_per_tile,
                        num_raster_channels,
                        -1);
          fill_col_halo(output_data,
                        tile_x,
                        tile_y,
                        x_tiles,
                        y_tiles,
                        halo_y_pixels_per_tile_boundary,
                        x_pixels_per_tile,
                        y_pixels_per_tile,
                        num_raster_channels,
                        1);
        }
      }
    }
  }
  return std::make_pair(output_data, raster_info);
}

}  // namespace RasterFormat_Namespace

#endif  // #ifdef __CUDACC__