/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseDataTableGpuResources.h"

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interop/QueryBuffer.h"
#include "QueryRenderer/PerGpuData.h"

namespace QueryRenderer {
//
// class LineDataTablePerGpuData
//
class LineDataTablePerGpuData : public BasePerGpuData {
 public:
  QueryVertexBufferShPtr vbo;
  QueryIndexBufferShPtr ibo;
  QueryShaderStorageBufferShPtr ssbo;
  QueryIndirectVboShPtr ind_vbo;
  QueryIndirectIboShPtr ind_ibo;

  explicit LineDataTablePerGpuData(RootPerGpuData& root_data,
                                   const QueryVertexBufferShPtr& vbo = nullptr,
                                   const QueryIndexBufferShPtr& ibo = nullptr,
                                   const QueryShaderStorageBufferShPtr& ssbo = nullptr,
                                   const QueryIndirectVboShPtr& ind_vbo = nullptr,
                                   const QueryIndirectIboShPtr& ind_ibo = nullptr);

  ~LineDataTablePerGpuData() override = default;

  void resetPointers();
};

//
// class LineDataTableGpuResources
//
class LineDataTableGpuResources
    : public BaseDataTableGpuResources<LineDataTablePerGpuData> {
 public:
  explicit LineDataTableGpuResources(DataInputFormat input_format);
  ~LineDataTableGpuResources() override = default;

  // Local methods
  void initGpuResourcesFromBuffers(const GlobalRenderContext& global_ctx,
                                   bool use_index_buffer = false,
                                   const std::string& vega_data_table_name = "");

  const gfx::IndexBuffer* getIndexBuffer(const GpuId gpu_id) const;
  const gfx::BufferWrapper* getShaderStorageBuffer(const GpuId gpu_id) const;
  const gfx::IndirectDrawVertexBuffer* getIndirectDrawVertexBuffer(
      const GpuId gpu_id) const;
  const gfx::IndirectDrawIndexBuffer* getIndirectDrawIndexBuffer(
      const GpuId gpu_id) const;

  // Clear the pointers in the gpu data map elements
  void resetDataPointers();

 private:
  DataInputFormat input_format_;
};

};  // namespace QueryRenderer
