/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Cache/RasterMeshMgr.h"

#include "CudaMgr/CudaMgr.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/RenderCmdQueue.h"

namespace QueryRenderer {

namespace {

RasterMeshBufferPtrs create_raster_mesh_buffers(
    const std::string& data_table_name,
    const RasterMeshMgr::BufferSizes& raster_mesh_buffer_sizes,
    RootPerGpuData& gpu_data) {
  RasterMeshBufferPtrs buffer_ptrs;

  auto& tmp_mesh_buffer_wk_ptrs =
      gpu_data.createTmpRasterMeshBuffersForDataTable(data_table_name);

  if (raster_mesh_buffer_sizes.vbo_bytes > 0) {
    tmp_mesh_buffer_wk_ptrs.vbo =
        gpu_data.getVboBufferPool().getInactiveRsrc(raster_mesh_buffer_sizes.vbo_bytes);
    buffer_ptrs.vbo = tmp_mesh_buffer_wk_ptrs.vbo.lock();
    CHECK(buffer_ptrs.vbo);
  }

  if (raster_mesh_buffer_sizes.ibo_bytes > 0) {
    tmp_mesh_buffer_wk_ptrs.ibo =
        gpu_data.getIboBufferPool().getInactiveRsrc(raster_mesh_buffer_sizes.ibo_bytes);
    buffer_ptrs.ibo = tmp_mesh_buffer_wk_ptrs.ibo.lock();
    CHECK(buffer_ptrs.ibo);
  }

  return buffer_ptrs;
}

inline void internal_set_cuda_context(CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                      const size_t gpu_index) {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr);
  cuda_mgr->setContext(gpu_index);
#endif  // HAVE_CUDA
}

}  // namespace

RasterMeshMgr::RasterMeshMgr(GlobalRenderContext& global_context,
                             RenderCmdQueue& command_queue,
                             CudaMgr_Namespace::CudaMgr* cuda_mgr)
    : global_context_{global_context}
    , command_queue_{command_queue}
    , cuda_mgr_{cuda_mgr} {}

RasterMeshBufferPtrs RasterMeshMgr::createRasterMeshBuffers(
    const std::string& poly_table_name,
    const size_t gpu_index,
    const BufferSizes& buffer_sizes) {
  RasterMeshBufferPtrs rtn;

  // creating GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    auto& gpu_data = global_context_.getGpuDataFromIndex(gpu_index);
    rtn = create_raster_mesh_buffers(poly_table_name, buffer_sizes, gpu_data);
  });

  return rtn;
}

RasterMeshBufferMemoryDescriptors RasterMeshMgr::getBufferDescriptors(
    const RasterMeshBufferPtrs& buffers,
    const size_t gpu_index) {
  RasterMeshBufferMemoryDescriptors rtn;
  // accessing GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    if (buffers.vbo) {
      rtn.vbo_descriptor = buffers.vbo->getBufferMemoryDescriptor();
    }
    if (buffers.ibo) {
      rtn.ibo_descriptor = buffers.ibo->getBufferMemoryDescriptor();
    }
  });
  return rtn;
}

}  // namespace QueryRenderer
