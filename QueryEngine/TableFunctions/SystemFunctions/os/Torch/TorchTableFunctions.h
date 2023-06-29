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

#include "QueryEngine/heavydbTypes.h"

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: tf_torch_raster_obj_detect__cpu_template(TableFunctionManager, Cursor<Column<PixelType> x,
   Column<PixelType> y, ColumnList<ColorType> channels>,
   PixelType x_input_units_per_pixel, PixelType y_input_units_per_pixel, float max_color_val,
   int64_t tile_boundary_halo_pixels, TextEncodingNone model_metadata_path, TextEncodingNone metadata_path, float min_confidence_threshold,
   float iou_threshold, bool use_gpu, int64_t device_num) | filter_table_function_transpose=on ->
   Column<TextEncodingDict> detected_class | input_id=args<>, Column<int32_t> detected_class_id, Column<double> x,
   Column<double> y, Column<double> detected_width, Column<double> detected_height,
   Column<float> detected_confidence, PixelType=[float, double],
   ColorType=[int16_t, int32_t]
 */
// clang-format on

template <typename PixelType, typename ColorType>
TEMPLATE_NOINLINE int32_t
tf_torch_raster_obj_detect__cpu_template(TableFunctionManager& mgr,
                                         const Column<PixelType>& input_x,
                                         const Column<PixelType>& input_y,
                                         const ColumnList<ColorType>& input_channels,
                                         const PixelType x_input_units_per_pixel,
                                         const PixelType y_input_units_per_pixel,
                                         const float max_color_value,
                                         const int64_t tile_boundary_halo_pixels,
                                         const TextEncodingNone& model_path,
                                         const TextEncodingNone& model_metadata_path,
                                         const float min_confidence_threshold,
                                         const float iou_threshold,
                                         const bool use_gpu,
                                         const int64_t device_num,
                                         Column<TextEncodingDict>& detected_class_label,
                                         Column<int32_t>& detected_class_id,
                                         Column<double>& detected_centroid_x,
                                         Column<double>& detected_centroid_y,
                                         Column<double>& detected_width,
                                         Column<double>& detected_height,
                                         Column<float>& detected_confidence);

#endif  // __CUDACC__