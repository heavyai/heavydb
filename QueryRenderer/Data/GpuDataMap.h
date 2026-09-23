/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "GfxDriver/RenderError.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

//
// template class GpuDataMap
//
// Maps GpuIds to generic GpuDataTypes
// GpuDataType currently has no requirements or contraints other than constructable
// It is *assumed* to inherit from QueryRenderer::BasePerGpuData
//
// TODO(scb): c++20 requires-expression is_base_of
//
template <typename GpuDataType>
class GpuDataMap {
 public:
  bool isEmpty() const { return gpu_data_map_.empty(); }

  bool hasData(GpuId gpu_id) const { return gpu_data_map_.count(gpu_id) > 0; }

  // Get cached vector of GpuIds
  // Does not validate that data stored is valid (e.g. vars may be null)
  const std::vector<GpuId>& getGpuIds() const { return gpu_ids_; }

  // Get a copy of GpuIds using predicate function to filter invalid elements
  using UsedGpuPredicate = std::function<bool(GpuId, const GpuDataType&)>;
  std::vector<GpuId> getUsedGpuIds(UsedGpuPredicate validate_cb) const {
    std::vector<GpuId> rtn;
    for (auto& [gpu_id, gpu_data] : gpu_data_map_) {
      if (validate_cb(gpu_id, gpu_data)) {
        rtn.emplace_back(gpu_id);
      }
    }
    return rtn;
  }

  // Get data for a specific gpu
  // Throws an error if GpuId is not found in map
  GpuDataType& getData(GpuId gpu_id) const {
    auto itr = gpu_data_map_.find(gpu_id);
    RUNTIME_EX_ASSERT(itr != gpu_data_map_.end(),
                      "Cannot find per gpu data for gpu " + std::to_string(gpu_id) +
                          ". Used gpus: " + to_string(gpu_ids_));
    return itr->second;
  }

  // Get the first data element in the map
  // returns nullptr if not found
  GpuDataType* getFirstData() const {
    if (auto itr = gpu_data_map_.begin(); itr != gpu_data_map_.end()) {
      return &itr->second;
    } else {
      return nullptr;
    }
  }

  // Visit each element in the map
  // Visitor callback must return true to continue processing
  using GpuDataVisitorCB = std::function<bool(GpuId, GpuDataType&)>;
  void visitData(GpuDataVisitorCB visitor) const {
    for (auto& [gpu_id, gpu_data] : gpu_data_map_) {
      if (!visitor(gpu_id, gpu_data)) {
        return;
      }
    }
  }

  // Emplace an element in the map
  // Requires GpuId for key, then forwards remaining arguments to map::try_emplace
  template <class... Args>
  std::pair<typename std::map<GpuId, GpuDataType>::iterator, bool> try_emplace(
      const GpuId gpu_id,
      Args&&... args) {
    auto rtn = gpu_data_map_.try_emplace(gpu_id, std::forward<decltype(args)>(args)...);
    if (rtn.second) {
      gpu_ids_.emplace_back(gpu_id);
    }
    return rtn;
  }

  // Erase vector of GpuIds from map
  void erase(const std::vector<GpuId>& gpu_ids) {
    for (auto gpu_id : gpu_ids) {
      gpu_data_map_.erase(gpu_id);
    }
    gpu_ids_.clear();
    for (auto& itr : gpu_data_map_) {
      gpu_ids_.emplace_back(itr.first);
    }
  }

  // clear contents
  void clear() {
    gpu_data_map_.clear();
    gpu_ids_.clear();
  }

 private:
  mutable std::map<GpuId, GpuDataType> gpu_data_map_;
  mutable std::vector<GpuId> gpu_ids_;
};

}  // namespace QueryRenderer
