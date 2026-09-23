/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__

#include "QueryEngine/TableFunctions/SystemFunctions/Shared/CpuTimer.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/Shared/RasterInfo.h"

#include <string>
#include <vector>

struct Detection {
  int32_t class_idx;
  std::string class_label;
  double centroid_x;
  double centroid_y;
  double width;
  double height;
  float confidence;
};

struct BoxDetection {
  double tl_x;
  double tl_y;
  double br_x;
  double br_y;
  float score;
  int class_idx;
};

struct ModelInfo {
  bool is_valid{false};
  int64_t batch_size{-1};
  int64_t raster_channels{-1};
  int64_t raster_tile_width{-1};
  int64_t raster_tile_height{-1};
  int64_t stride{-1};
  std::vector<std::string> class_labels;
};

ModelInfo get_model_info_from_file(const std::string& filename);

std::vector<Detection> detect_objects_in_tiled_raster(
    const std::string& model_path,
    const ModelInfo& model_info,
    const bool use_gpu,
    const int64_t device_num,
    std::vector<float>& raster_data,
    const RasterFormat_Namespace::RasterInfo& raster_info,
    const float min_confidence_threshold,
    const float iou_threshold,
    std::shared_ptr<CpuTimer> timer);

class TorchWarmer {
 public:
  static bool warmup_torch(const std::string& model_path,
                           const bool use_gpu,
                           const int64_t device_num);
  static bool is_torch_warmed;
};

#endif  // __CUDACC__