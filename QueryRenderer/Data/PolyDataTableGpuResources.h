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
// class PolyDataTablePerGpuData
//
class PolyDataTablePerGpuData : public BasePerGpuData {
 public:
  QueryVertexBufferShPtr vbo;
  QueryShaderStorageBufferShPtr ssbo;
  QueryIndirectVboShPtr ivbo_lines;
  QueryIndirectVboShPtr ivbo_polys;
  PolyDrawBatchInfoUqPtr poly_draw_batch_info;
  QueryShaderStorageBufferShPtr poly_rowids;

  explicit PolyDataTablePerGpuData(
      RootPerGpuData& rootData,
      const QueryVertexBufferShPtr& vbo = nullptr,
      const QueryShaderStorageBufferShPtr& ssbo = nullptr,
      const QueryIndirectVboShPtr& ivbo_lines = nullptr,
      const QueryIndirectVboShPtr& ivbo_polys = nullptr,
      const QueryShaderStorageBufferShPtr& poly_rowids = nullptr);

  ~PolyDataTablePerGpuData() override = default;

  void resetPointers();
};

//
// class PolyDataTableGpuResources
//
class PolyDataTableGpuResources
    : public BaseDataTableGpuResources<PolyDataTablePerGpuData> {
 public:
  explicit PolyDataTableGpuResources(DataInputFormat input_format);
  ~PolyDataTableGpuResources() override = default;

  // local methods
  void initGpuResourcesFromBuffers(const GlobalRenderContext& global_ctx,
                                   const std::string& vega_data_table_name);

  const gfx::IndirectDrawVertexBuffer* getIndirectDrawVertexBuffer_lines(
      const GpuId gpu_id) const;
  const gfx::IndirectDrawVertexBuffer* getIndirectDrawVertexBuffer_polys(
      const GpuId gpu_id) const;
  const gfx::BufferWrapper* getShaderStorageBuffer_poly_rowids(const GpuId gpu_id) const;

  // Clear the pointers in the gpu data map elements
  void resetDataPointers();

  void setPolyDrawBatchInfo(const GpuId gpu_id,
                            PolyDrawBatchInfoUqPtr&& poly_draw_batch_info);
  const PolyDrawBatchInfo& getPolyDrawBatchInfo(const GpuId gpu_id) const;

 private:
  DataInputFormat input_format_;
};

}  // namespace QueryRenderer
