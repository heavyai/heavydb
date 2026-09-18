/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/GpuDataMap.h"
#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

//
// template class BaseDataTableGpuResources
//
// Wraps a GpuDataMap providing common functions used by DataTable implementations
// GpuDataType requirements:
//  - inherits from BasePerGpuData
//  - has QueryVertexBufferShPtr member named 'vbo'
//
// TODO(scb): enforce using concepts and requires-expressions (c++20)
//
template <typename GpuDataType>
class BaseDataTableGpuResources {
 public:
  virtual ~BaseDataTableGpuResources() = default;

  const GpuDataMap<GpuDataType>& getGpuDataMap() const { return gpu_data_map_; }

  // Check if any GpuDataType stores a vertex buffer with > 0 vertices
  // Passing nullptr for data_layout will retrieve a singleton layout from
  // BufferLayoutManager
  // See BufferLayoutManager::getBufferLayoutDataToUse
  bool hasVerticesForLayout(const QueryDataLayoutShPtr& data_layout) const {
    bool did_find_data = false;
    gpu_data_map_.visitData([&](GpuId gpu_id, GpuDataType& gpu_data) {
      if (gpu_data.vbo && gpu_data.vbo->numVertices(data_layout)) {
        did_find_data = true;
        return false;  // stop visitor
      }
      return true;
    });
    return did_find_data;
  }

  // TODO(scb): rename to getDataVertexBuffers or just getVertexBuffers?
  std::map<GpuId, QueryLayoutBufferWkPtr> getDataBuffers() const {
    std::map<GpuId, QueryLayoutBufferWkPtr> rtn;
    gpu_data_map_.visitData([&](GpuId gpu_id, GpuDataType& gpu_data) {
      rtn.emplace(gpu_id, gpu_data.vbo);
      return true;
    });
    return rtn;
  }

  // Check if any of the GpuDataType elements has a vertex buffer for
  // the given layout, which is sufficient to determine if there is
  // result data stored. Derived classes can perform further checks
  // TODO(scb): require a isComplete() function for GpuDataType to check
  // all required fields?
  bool hasDataForLayout(const QueryDataLayoutShPtr& layout) const {
    bool did_find_data = true;  // assume ok
    gpu_data_map_.visitData([&](GpuId gpu_id, GpuDataType& gpu_data) {
      if (gpu_data.vbo && !gpu_data.vbo->hasBufferLayout(*layout)) {
        RENDER_LOG() << "missing buffer layout for gpu: " << gpu_id;
        did_find_data = false;
        return false;
      }
      return true;
    });
    return did_find_data;
  }

 protected:
  GpuDataMap<GpuDataType> gpu_data_map_;
};

}  // namespace QueryRenderer
