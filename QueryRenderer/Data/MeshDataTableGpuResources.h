/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseDataTableGpuResources.h"

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/PerGpuData.h"

namespace QueryRenderer {

//
// class MeshDataTablePerGpuData
//
class MeshDataTablePerGpuData : public BasePerGpuData {
 public:
  QueryVertexBufferShPtr vbo;
  QueryIndexBufferShPtr ibo;

  explicit MeshDataTablePerGpuData(RootPerGpuData& root_data,
                                   QueryVertexBufferShPtr in_vbo = nullptr,
                                   QueryIndexBufferShPtr in_ibo = nullptr);

  ~MeshDataTablePerGpuData() override;

  void clear();
};

//
// class MeshDataTableGpuResources
//
class MeshDataTableGpuResources
    : public BaseDataTableGpuResources<MeshDataTablePerGpuData> {
 public:
  explicit MeshDataTableGpuResources(DataInputFormat input_format)
      : BaseDataTableGpuResources(), input_format_{input_format} {}
  ~MeshDataTableGpuResources() override = default;

  void initGpuResourcesFromBuffers(const GlobalRenderContext& global_ctx,
                                   const std::string& vega_data_table_name = "");

 private:
  DataInputFormat input_format_;
};

}  // namespace QueryRenderer
