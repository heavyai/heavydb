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

#ifndef __CUDACC__

#include "TorchTableFunctions.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/CpuTimer.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/RasterFormat.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Torch/TorchWrapper.h"

#include <cstdio>

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
                                         Column<float>& detected_confidence) {
  std::shared_ptr<CpuTimer> timer =
      std::make_shared<CpuTimer>("tf_torch_raster_obj_detect");

  timer->start_event_timer("get_model_info_from_file");
  const std::string model_path_str(model_path.getString());
  std::string model_metadata_path_str(model_metadata_path.getString());
  if (model_metadata_path_str.empty()) {
    model_metadata_path_str = model_path_str;
  }
  const auto model_info = get_model_info_from_file(model_metadata_path_str);
  if (!model_info.is_valid) {
    return mgr.ERROR_MESSAGE("Could not get model info from file.");
  }
  if (model_info.class_labels.empty()) {
    return mgr.ERROR_MESSAGE("Could not get class labels from file.");
  }
  const auto class_idx_to_label_vec =
      detected_class_label.string_dict_proxy_->getOrAddTransientBulk(
          model_info.class_labels);
  const int64_t num_class_labels = static_cast<int64_t>(class_idx_to_label_vec.size());

  constexpr int64_t target_batch_size_multiple{8};

  auto raster_data =
      RasterFormat_Namespace::format_raster_data<PixelType, ColorType, float>(
          input_x,
          input_y,
          input_channels,
          x_input_units_per_pixel,
          y_input_units_per_pixel,
          max_color_value,
          model_info.raster_tile_width,
          model_info.raster_tile_height,
          tile_boundary_halo_pixels,
          target_batch_size_multiple,
          timer->start_nested_event_timer("format_raster_data"));

  try {
    const auto processed_detections = detect_objects_in_tiled_raster(
        model_path_str,
        model_info,
        use_gpu,
        device_num,
        raster_data.first,
        raster_data.second,
        min_confidence_threshold,
        iou_threshold,
        timer->start_nested_event_timer("detect_objects_in_tiled_raster"));
    timer->start_event_timer("Write results");
    const int64_t num_detections = processed_detections.size();
    mgr.set_output_row_size(num_detections);
    // The class labels taken from the model file will be in same order as class idxs by
    // definition
    for (int64_t detection_idx = 0; detection_idx < num_detections; ++detection_idx) {
      const auto class_idx = processed_detections[detection_idx].class_idx;
      detected_class_id[detection_idx] = class_idx;
      if (class_idx < 0 || class_idx >= num_class_labels) {
        detected_class_label.setNull(detection_idx);
      } else {
        detected_class_label[detection_idx] = class_idx_to_label_vec[class_idx];
      }
      detected_centroid_x[detection_idx] = processed_detections[detection_idx].centroid_x;
      detected_centroid_y[detection_idx] = processed_detections[detection_idx].centroid_y;
      detected_width[detection_idx] = processed_detections[detection_idx].width;
      detected_height[detection_idx] = processed_detections[detection_idx].height;
      detected_confidence[detection_idx] = processed_detections[detection_idx].confidence;
    }
    return num_detections;
  } catch (const std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  return 0;
}

template TEMPLATE_NOINLINE int32_t
tf_torch_raster_obj_detect__cpu_template(TableFunctionManager& mgr,
                                         const Column<float>& input_x,
                                         const Column<float>& input_y,
                                         const ColumnList<int16_t>& input_channels,
                                         const float x_input_units_per_pixel,
                                         const float y_input_units_per_pixel,
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

template TEMPLATE_NOINLINE int32_t
tf_torch_raster_obj_detect__cpu_template(TableFunctionManager& mgr,
                                         const Column<float>& input_x,
                                         const Column<float>& input_y,
                                         const ColumnList<int32_t>& input_channels,
                                         const float x_input_units_per_pixel,
                                         const float y_input_units_per_pixel,
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

template TEMPLATE_NOINLINE int32_t
tf_torch_raster_obj_detect__cpu_template(TableFunctionManager& mgr,
                                         const Column<double>& input_x,
                                         const Column<double>& input_y,
                                         const ColumnList<int16_t>& input_channels,
                                         const double x_input_units_per_pixel,
                                         const double y_input_units_per_pixel,
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

template TEMPLATE_NOINLINE int32_t
tf_torch_raster_obj_detect__cpu_template(TableFunctionManager& mgr,
                                         const Column<double>& input_x,
                                         const Column<double>& input_y,
                                         const ColumnList<int32_t>& input_channels,
                                         const double x_input_units_per_pixel,
                                         const double y_input_units_per_pixel,
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

#endif