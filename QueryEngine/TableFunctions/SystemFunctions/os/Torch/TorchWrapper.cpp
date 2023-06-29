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

#include "TorchWrapper.h"
#include "Shared/funcannotations.h"
#include "TorchOps.hpp"

#undef GLOBAL
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#ifdef HAVE_CUDA_TORCH
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "rapidjson/document.h"

std::string get_device_string(const bool use_gpu, const int64_t device_num) {
  std::string device_type{"cpu"};
#ifdef HAVE_CUDA_TORCH
  if (torch::cuda::is_available() && use_gpu) {
    device_type = "cuda:" + std::to_string(device_num);
  }
#endif
  return device_type;
}

bool should_use_half(const bool use_gpu, const std::string& model_path) {
  bool use_half = false;
#ifdef HAVE_CUDA_TORCH
  if (use_gpu && model_path.find("half") != std::string::npos) {
    use_half = true;
  }
#endif
  return use_half;
}

static std::unordered_map<std::string, std::shared_ptr<torch::jit::script::Module>>
    model_cache;
static std::shared_mutex model_mutex;

std::shared_ptr<torch::jit::script::Module> get_model_from_cache(
    const std::string& model_path) {
  std::shared_lock<std::shared_mutex> model_cache_read_lock(model_mutex);
  auto model_itr = model_cache.find(model_path);
  if (model_itr == model_cache.end()) {
    return nullptr;
  }
  return model_itr->second;
}

void add_model_to_cache(const std::string& model_path,
                        std::shared_ptr<torch::jit::script::Module> model_module) {
  std::unique_lock<std::shared_mutex> model_cache_write_lock(model_mutex);
  model_cache.emplace(model_path, model_module);
}

std::shared_ptr<torch::jit::script::Module> load_module(const std::string& model_path,
                                                        const std::string compute_device,
                                                        const at::ScalarType data_type,
                                                        // const bool use_half,
                                                        const bool use_cache) {
  std::shared_ptr<torch::jit::script::Module> module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    if (use_cache) {
      module = get_model_from_cache(model_path);
    }
    if (module == nullptr) {  // module not found or not using cache
      module = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
      module->to(compute_device, data_type);
      module->eval();

      if (use_cache) {
        add_model_to_cache(model_path, module);
      }
    } else {
      module->eval();
    }
  } catch (const c10::Error& e) {
    std::string error_msg{"Error loading the provided model: "};
    error_msg += e.what();
    throw std::runtime_error(error_msg);
  }
  return module;
}

std::string get_json_str_from_file_header(const std::string& filename,
                                          const size_t max_search_chars) {
  std::ifstream model_file(filename);
  bool found_opening_brace = false;
  size_t brace_nest_count = 0;
  size_t char_idx{0};
  std::string json_str;
  if (model_file.is_open()) {
    char c;
    while (model_file.get(c) && (brace_nest_count >= 1 || char_idx < max_search_chars)) {
      char_idx++;
      if (c == '{') {
        found_opening_brace = true;
        brace_nest_count++;
      }
      if (found_opening_brace) {
        json_str += c;
      }
      if (c == '}') {
        if (brace_nest_count > 0) {
          brace_nest_count--;
          if (found_opening_brace &&
              brace_nest_count == 0) {  // found_opening_brace superfluous
            break;
          }
        }
      }
    }
  }
  if (found_opening_brace && brace_nest_count == 0) {
    return json_str;
  }
  return "";
}

ModelInfo get_model_info_from_json(const std::string& json_str) {
  ModelInfo model_info;
  rapidjson::Document doc;
  if (doc.Parse<0>(json_str.c_str()).HasParseError()) {
    return model_info;  // will have is_valid set to false
  }
  const auto shape_array_itr = doc.FindMember("shape");
  if (shape_array_itr != doc.MemberEnd() && shape_array_itr->value.IsArray()) {
    const rapidjson::SizeType num_shape_elems = shape_array_itr->value.Size();
    if (num_shape_elems == 4) {
      model_info.batch_size = shape_array_itr->value[0].GetInt();
      model_info.raster_channels = shape_array_itr->value[1].GetInt();
      model_info.raster_tile_height = shape_array_itr->value[2].GetInt();
      model_info.raster_tile_width = shape_array_itr->value[3].GetInt();
    }
  }
  const auto stride_itr = doc.FindMember("stride");
  if (stride_itr != doc.MemberEnd() && stride_itr->value.IsInt()) {
    model_info.stride = stride_itr->value.GetInt();
  }
  const auto class_labels_itr = doc.FindMember("names");
  if (class_labels_itr != doc.MemberEnd() && class_labels_itr->value.IsArray()) {
    const rapidjson::SizeType num_class_labels = class_labels_itr->value.Size();
    model_info.class_labels.reserve(static_cast<size_t>(num_class_labels));
    for (auto& label : class_labels_itr->value.GetArray()) {
      model_info.class_labels.emplace_back(label.GetString());
    }
  }
  model_info.is_valid = true;
  return model_info;
}

enum DetectionIdx {
  centroid_x = 0,
  centroid_y = 1,
  width = 2,
  height = 3,
  class_idx = 4,
  score = 5
};

enum BoxDetectionIdx {
  tl_x = 0,
  tl_y = 1,
  br_x = 2,
  br_y = 3,
};

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
  auto y = torch::zeros_like(x);
  // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2,
  // y2)
  y.select(1, BoxDetectionIdx::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
  y.select(1, BoxDetectionIdx::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
  y.select(1, BoxDetectionIdx::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
  y.select(1, BoxDetectionIdx::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
  return y;
}

torch::Tensor world_scale_detections(
    const torch::Tensor& input,
    const int64_t batch_idx,
    const RasterFormat_Namespace::RasterInfo& raster_info) {
  const int64_t tile_y_idx = batch_idx / raster_info.x_tiles;
  const int64_t tile_x_idx = batch_idx % raster_info.x_tiles;
  const double tile_x0_pixel = tile_x_idx * raster_info.logical_x_pixels_per_tile -
                               raster_info.halo_x_pixels_per_tile_boundary;
  const double tile_y0_pixel = tile_y_idx * raster_info.logical_y_pixels_per_tile -
                               raster_info.halo_y_pixels_per_tile_boundary;
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  //.device(torch::kCPU);

  auto output = torch::zeros_like(input, options);
  output.select(1, DetectionIdx::centroid_x) =
      (input.select(1, DetectionIdx::centroid_x) + tile_x0_pixel) *
          raster_info.x_input_units_per_pixel +
      raster_info.min_x_input;
  output.select(1, DetectionIdx::centroid_y) =
      (input.select(1, DetectionIdx::centroid_y) + tile_y0_pixel) *
          raster_info.y_input_units_per_pixel +
      raster_info.min_y_input;
  output.select(1, DetectionIdx::width) =
      input.select(1, DetectionIdx::width) * raster_info.x_input_units_per_pixel;
  output.select(1, DetectionIdx::height) =
      input.select(1, DetectionIdx::height) * raster_info.y_input_units_per_pixel;
  output.select(1, DetectionIdx::class_idx) = input.select(1, DetectionIdx::class_idx);
  output.select(1, DetectionIdx::score) = input.select(1, DetectionIdx::score);
  return output;
}

std::vector<Detection> process_detections(
    const torch::Tensor& raw_detections,
    const float min_confidence_threshold,
    const float iou_threshold,
    const ModelInfo& model_info,
    const RasterFormat_Namespace::RasterInfo& raster_info,
    std::shared_ptr<CpuTimer> timer) {
  // Most of logic in this function borrowed liberally from
  // https://github.com/yasenh/libtorch-yolov5 (MIT Licensed)
  timer->start_event_timer("Confidence mask");
  constexpr int64_t item_attr_size = 5;
  const auto& class_labels = model_info.class_labels;
  const int32_t num_class_labels = static_cast<int32_t>(class_labels.size());
  const auto batch_size = raster_info.x_tiles * raster_info.y_tiles;
  const auto num_classes = raw_detections.size(2) - item_attr_size;
  auto conf_mask = raw_detections.select(2, 4).ge(min_confidence_threshold).unsqueeze(2);
  torch::Tensor all_world_scaled_detections;
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto masked_detections =
        torch::masked_select(raw_detections[batch_idx], conf_mask[batch_idx])
            .view({-1, num_classes + item_attr_size});

    if (masked_detections.size(0) == 0) {
      continue;
    }
    // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
    masked_detections.slice(1, item_attr_size, item_attr_size + num_classes) *=
        masked_detections.select(1, 4).unsqueeze(1);

    // [best class only] get the max classes score at each result (e.g. elements 5-84)
    std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(
        masked_detections.slice(1, item_attr_size, item_attr_size + num_classes), 1);

    // class score
    auto max_conf_scores = std::get<0>(max_classes);
    // index
    auto max_conf_classes = std::get<1>(max_classes);

    max_conf_scores = max_conf_scores.to(torch::kFloat).unsqueeze(1);
    max_conf_classes = max_conf_classes.to(torch::kFloat).unsqueeze(1);
    masked_detections = torch::cat(
        {masked_detections.slice(1, 0, 4), max_conf_classes, max_conf_scores}, 1);

    if (raster_info.halo_x_pixels_per_tile_boundary > 0 ||
        raster_info.halo_y_pixels_per_tile_boundary > 0) {
      const double min_x_pixel = raster_info.halo_x_pixels_per_tile_boundary;
      const double max_x_pixel =
          raster_info.x_pixels_per_tile - 1 - raster_info.halo_x_pixels_per_tile_boundary;
      const double min_y_pixel = raster_info.halo_y_pixels_per_tile_boundary;
      const double max_y_pixel =
          raster_info.y_pixels_per_tile - 1 - raster_info.halo_y_pixels_per_tile_boundary;
      auto x_halo_mask =
          torch::logical_and(masked_detections.select(1, 0).ge(min_x_pixel),
                             masked_detections.select(1, 0).le(max_x_pixel));
      auto y_halo_mask =
          torch::logical_and(masked_detections.select(1, 1).ge(min_y_pixel),
                             masked_detections.select(1, 1).le(max_y_pixel));

      auto halo_mask = torch::logical_and(x_halo_mask, y_halo_mask).unsqueeze(1);
      masked_detections =
          torch::masked_select(masked_detections, halo_mask).view({-1, 6});
    }

    auto world_scaled_detections =
        world_scale_detections(masked_detections, batch_idx, raster_info);

    auto world_scaled_detections_cpu = world_scaled_detections.cpu();
    if (batch_idx == 0) {
      all_world_scaled_detections = world_scaled_detections_cpu.cpu();
    } else {
      all_world_scaled_detections =
          torch::cat({all_world_scaled_detections, world_scaled_detections_cpu}, 0).cpu();
    }
  }
  timer->start_event_timer("Per-batch processing");
  std::vector<Detection> processed_detections;
  if (all_world_scaled_detections.size(0) == 0) {
    return processed_detections;
  }

  torch::Tensor bboxes = xywh2xyxy(all_world_scaled_detections.slice(1, 0, 4));

  auto kept_bboxes_idxs =
      nms_kernel(bboxes, all_world_scaled_detections.select(1, 5), iou_threshold);

  timer->start_event_timer("Nms processing");

  const int64_t num_kept_detections = kept_bboxes_idxs.size(0);
  processed_detections.reserve(num_kept_detections);

  const auto& kept_bboxes_idxs_accessor = kept_bboxes_idxs.accessor<int64_t, 1>();
  const auto& detections_array_accessor =
      all_world_scaled_detections.accessor<double, 2>();

  for (int64_t detection_idx = 0; detection_idx < num_kept_detections; ++detection_idx) {
    int64_t kept_detection_idx = kept_bboxes_idxs_accessor[detection_idx];
    const auto& detection_array = detections_array_accessor[kept_detection_idx];
    const int32_t class_idx =
        static_cast<int32_t>(round(detection_array[DetectionIdx::class_idx]));
    std::string class_label;
    if (class_idx < num_class_labels) {
      class_label = class_labels[class_idx];
    }
    Detection processed_detection{
        class_idx,
        class_label,
        detection_array[DetectionIdx::centroid_x],
        detection_array[DetectionIdx::centroid_y],
        detection_array[DetectionIdx::width],
        detection_array[DetectionIdx::height],
        static_cast<float>(detection_array[DetectionIdx::score])};
    processed_detections.emplace_back(processed_detection);
  }
  timer->start_event_timer("Output processing");
  return processed_detections;
}

std::vector<Detection> detect_objects_in_tiled_raster_impl(
    const std::string& model_path,
    const ModelInfo& model_info,
    const bool use_gpu,
    const int64_t device_num,
    std::vector<float>& raster_data,
    const RasterFormat_Namespace::RasterInfo& raster_info,
    const float min_confidence_threshold,
    const float iou_threshold,
    std::shared_ptr<CpuTimer> timer) {
  const std::string compute_device = get_device_string(use_gpu, device_num);
  const bool use_half = should_use_half(use_gpu, model_path);
  const auto input_data_type = use_half ? torch::kHalf : torch::kFloat32;
  const bool use_model_cache = use_gpu;

  try {
    // Moved try block up to here as InferenceMode call below can throw if GPU is not
    // specified correctly
#ifdef HAVE_CUDA_TORCH
    c10::cuda::OptionalCUDAGuard cuda_guard;
    if (use_gpu) {
      cuda_guard.set_index(static_cast<int8_t>(device_num));
    }
#endif

    c10::InferenceMode guard;
    torch::NoGradGuard no_grad;

    timer->start_event_timer("Model load");

    auto module =
        load_module(model_path, compute_device, input_data_type, use_model_cache);
    timer->start_event_timer("Input prep");
    std::cout << "Device: " << compute_device << " Use half: " << use_half << std::endl;

    std::cout << "X tiles: " << raster_info.x_tiles << " Y tiles: " << raster_info.y_tiles
              << " Batch size: " << raster_info.batch_tiles << std::endl;

    auto input_tensor =
        torch::from_blob(raster_data.data(),
                         {raster_info.batch_tiles,
                          raster_info.raster_channels,
                          raster_info.y_pixels_per_tile,
                          raster_info.x_pixels_per_tile} /*, tensor_options */)
            .to(compute_device, input_data_type);

    std::vector<torch::jit::IValue> module_input;
    module_input.emplace_back(input_tensor);

    timer->start_event_timer("Inference");
    torch::jit::IValue output = module->forward(module_input);

    auto raw_detections = output.toTuple()->elements()[0].toTensor();

#ifdef HAVE_CUDA_TORCH
    constexpr bool enable_debug_timing{true};
    if (enable_debug_timing && use_gpu) {
      std::cout << "Synchronizing timing" << std::endl;
      c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
      AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
#endif

    const auto processed_detections =
        process_detections(raw_detections,
                           min_confidence_threshold,
                           iou_threshold,
                           model_info,
                           raster_info,
                           timer->start_nested_event_timer("process_detections"));

    return processed_detections;

  } catch (std::exception& e) {
    std::string error_msg{"Error during model inference: "};
    error_msg += e.what();
    throw std::runtime_error(error_msg);
  }
}

void print_model_params(const std::string& model_path,
                        const bool use_gpu,
                        const int64_t device_num) {
  const std::string compute_device = get_device_string(use_gpu, device_num);
  const bool use_half = should_use_half(use_gpu, model_path);
  const auto input_data_type = use_half ? torch::kHalf : torch::kFloat32;

  try {
    auto module =
        load_module(model_path, compute_device, input_data_type, use_gpu /* use_cache */);
    const auto module_named_params = module->named_parameters(true);
    const size_t num_named_params = module_named_params.size();
    std::cout << "Module # params: " << num_named_params << std::endl;
    const size_t max_params_to_print{1000};
    size_t param_idx{0};
    for (const auto& param : module_named_params) {
      std::cout << param.name << std::endl;
      if (param_idx++ == max_params_to_print) {
        break;
      }
    }
    const auto module_named_buffers = module->named_buffers(true);
    const size_t num_named_buffers = module_named_buffers.size();
    std::cout << "Module # named buffers: " << num_named_buffers << std::endl;
    const auto module_named_children = module->named_children();
    const size_t num_named_children = module_named_children.size();
    std::cout << "Module # named children: " << num_named_children << std::endl;
    std::cout << "Finishing torch warmup" << std::endl;
  } catch (std::exception& e) {
    std::string error_msg{"Error fetching Torch model params: "};
    error_msg += e.what();
    std::cout << error_msg << std::endl;
  }
}

__attribute__((__used__)) ModelInfo get_model_info_from_file(
    const std::string& filename) {
  const std::string json_str =
      get_json_str_from_file_header(filename, 100 /* max_search_chars */);
  if (json_str.size() > 0) {
    const ModelInfo model_info = get_model_info_from_json(json_str);
    if (model_info.is_valid) {
      return model_info;
    }
  }
  return {};
}

__attribute__((__used__)) std::vector<Detection> detect_objects_in_tiled_raster(
    const std::string& model_path,
    const ModelInfo& model_info,
    const bool use_gpu,
    const int64_t device_num,
    std::vector<float>& raster_data,
    const RasterFormat_Namespace::RasterInfo& raster_info,
    const float min_confidence_threshold,
    const float iou_threshold,
    std::shared_ptr<CpuTimer> timer) {
  return detect_objects_in_tiled_raster_impl(model_path,
                                             model_info,
                                             use_gpu,
                                             device_num,
                                             raster_data,
                                             raster_info,
                                             min_confidence_threshold,
                                             iou_threshold,
                                             timer);
}

bool TorchWarmer::warmup_torch(const std::string& model_path,
                               const bool use_gpu,
                               const int64_t device_num) {
  const auto model_info = get_model_info_from_file(model_path);
  for (size_t l = 0; l < model_info.class_labels.size(); ++l) {
    std::cout << l << ": " << model_info.class_labels[l] << std::endl;
  }
  print_model_params(model_path, use_gpu, device_num);
  return true;
}

// bool TorchWarmer::is_torch_warmed =
// TorchWarmer::warmup_torch("/home/todd/ml_models/xview_v4_latest.torchscript", true, 0);

#endif  // #ifndef __CUDACC__