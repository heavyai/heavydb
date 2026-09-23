/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "QueryRenderer/Types.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace QueryRenderer {

class RenderCmdQueue;
struct RasterMeshBufferPtrs;
struct RasterMeshBufferMemoryDescriptors;

class RasterMeshMgr : boost::noncopyable {
 public:
  struct BufferSizes {
    uint64_t vbo_bytes;
    uint64_t ibo_bytes;
  };

  RasterMeshMgr(GlobalRenderContext& global_context,
                RenderCmdQueue& command_queue,
                CudaMgr_Namespace::CudaMgr* cuda_mgr);

  RasterMeshBufferPtrs createRasterMeshBuffers(const std::string& poly_table_name,
                                               const size_t gpu_index,
                                               const BufferSizes& buffer_sizes);

  RasterMeshBufferMemoryDescriptors getBufferDescriptors(
      const RasterMeshBufferPtrs& buffers,
      const size_t gpu_index);

 private:
  GlobalRenderContext& global_context_;
  RenderCmdQueue& command_queue_;
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
};

}  // namespace QueryRenderer
