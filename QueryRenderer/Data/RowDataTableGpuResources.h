/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseDataTableGpuResources.h"

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/PerGpuData.h"

namespace QueryRenderer {

//
// class RowDataTablePerGpuData
//
class RowDataTablePerGpuData : public BasePerGpuData {
 public:
  QueryVertexBufferShPtr vbo;

  explicit RowDataTablePerGpuData(RootPerGpuData& root_data, QueryVertexBufferShPtr vbo)
      : BasePerGpuData(root_data), vbo(vbo) {}

  ~RowDataTablePerGpuData() override = default;
};

//
// class RowDataTableGpuResources
//
// GpuResources for unmodified row based result sets
// Used by point style renders
//
class RowDataTableGpuResources
    : public BaseDataTableGpuResources<RowDataTablePerGpuData> {
 public:
  explicit RowDataTableGpuResources(DataInputFormat input_format)
      : BaseDataTableGpuResources(), input_format_{input_format} {}
  ~RowDataTableGpuResources() override = default;

  // Local methods
  void initGpuResourcesFromBuffers(const GlobalRenderContext& global_ctx,
                                   const QueryDataLayoutShPtr& layout);

  bool hasLayouts() const;
  QueryDataLayoutShPtr getDataLayout() const;

  bool updateOffsetBytesMap(const QueryDataLayoutShPtr& data_layout);
  void clearOffsetBytesMap();

 private:
  DataInputFormat input_format_;
  std::map<GpuId, size_t> curr_buf_offset_bytes_;
};

}  // namespace QueryRenderer
