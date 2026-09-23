/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Cache/PolyMgr.h"

#include "CudaMgr/CudaMgr.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/RenderCmdQueue.h"

namespace QueryRenderer {
namespace {

PolyBufferPtrs create_the_poly_table_buffers(const std::string& data_table_name,
                                             const PolyTableByteData& table_byte_data,
                                             RootPerGpuData& gpu_data) {
  PolyBufferPtrs buffer_ptrs;

  auto& tmp_poly_buffer_wk_ptrs =
      gpu_data.createTmpPolyBuffersForDataTable(data_table_name);

  if (table_byte_data.num_vertex_bytes > 0) {
    tmp_poly_buffer_wk_ptrs.verts =
        gpu_data.getVboBufferPool().getInactiveRsrc(table_byte_data.num_vertex_bytes);
    buffer_ptrs.verts = tmp_poly_buffer_wk_ptrs.verts.lock();
    CHECK(buffer_ptrs.verts);
  }

  if (table_byte_data.num_line_indirect_draw_bytes > 0) {
    tmp_poly_buffer_wk_ptrs.line_draw_struct =
        gpu_data.getIndVboBufferPool().getInactiveRsrc(
            table_byte_data.num_line_indirect_draw_bytes);
    buffer_ptrs.line_draw_struct = tmp_poly_buffer_wk_ptrs.line_draw_struct.lock();
    CHECK(buffer_ptrs.line_draw_struct);
  }

  if (table_byte_data.num_poly_indirect_draw_bytes > 0) {
    tmp_poly_buffer_wk_ptrs.poly_draw_struct =
        gpu_data.getIndVboBufferPool().getInactiveRsrc(
            table_byte_data.num_poly_indirect_draw_bytes);
    buffer_ptrs.poly_draw_struct = tmp_poly_buffer_wk_ptrs.poly_draw_struct.lock();
    CHECK(buffer_ptrs.poly_draw_struct);
  }

  if (table_byte_data.num_ssbo_bytes > 0) {
    tmp_poly_buffer_wk_ptrs.per_row_data =
        gpu_data.getSsboBufferPool().getInactiveRsrc(table_byte_data.num_ssbo_bytes);
    buffer_ptrs.per_row_data = tmp_poly_buffer_wk_ptrs.per_row_data.lock();
    CHECK(buffer_ptrs.per_row_data);
  }

  if (table_byte_data.num_poly_rowids_bytes > 0) {
    tmp_poly_buffer_wk_ptrs.poly_rowids = gpu_data.getSsboBufferPool().getInactiveRsrc(
        table_byte_data.num_poly_rowids_bytes);
    buffer_ptrs.poly_rowids = tmp_poly_buffer_wk_ptrs.poly_rowids.lock();
    CHECK(buffer_ptrs.poly_rowids);
  }

  return buffer_ptrs;
}

void set_the_poly_table_poly_draw_batch_info(
    const std::string& data_table_name,
    PolyDrawBatchInfoUqPtr&& poly_draw_batch_info,
    RootPerGpuData& gpu_data) {
  gpu_data.createPolyDrawBatchInfoForDataTable(data_table_name,
                                               std::move(poly_draw_batch_info));
}

void set_the_poly_table_buffer_ready_for_render(const PolyBufferPtrs& poly_buffers,
                                                const QueryDataLayoutShPtr& vert_layout,
                                                const QueryDataLayoutShPtr& ssbo_layout) {
  // NOTE: it is assumed that all bytes allocated are used
  if (poly_buffers.verts && vert_layout) {
    poly_buffers.verts->setQueryDataLayout(
        vert_layout, poly_buffers.verts->getNumBytes(), 0);
  }

  if (poly_buffers.per_row_data && ssbo_layout) {
    poly_buffers.per_row_data->setQueryDataLayout(
        ssbo_layout, poly_buffers.per_row_data->getNumBytes(), 0);
  }
}

inline void internal_set_cuda_context(CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                      const size_t gpu_index) {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr);
  cuda_mgr->setContext(gpu_index);
#endif  // HAVE_CUDA
}

}  // namespace

PolyMgr::PolyMgr(GlobalRenderContext& global_context,
                 RenderCmdQueue& command_queue,
                 CudaMgr_Namespace::CudaMgr* cuda_mgr)
    : global_context_(global_context)
    , command_queue_(command_queue)
    , cuda_mgr_(cuda_mgr) {}

//
// in-situ polys
//

PolyBufferPtrs PolyMgr::createPolyTableInSituBuffers(
    const std::string& poly_table_name,
    const size_t gpu_index,
    const PolyTableByteData& init_table_data) {
  PolyBufferPtrs rtn;
  // creating GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    std::lock_guard<std::mutex> render_lock(insitu_poly_mutex_);
    auto& gpu_data = global_context_.getGpuDataFromIndex(gpu_index);
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    rtn = create_the_poly_table_buffers(poly_table_name, init_table_data, gpu_data);
  });
  return rtn;
}

PolyBufferMemoryDescriptors PolyMgr::getPolyTableInSituBufferDescriptors(
    const PolyBufferPtrs& buffers,
    const size_t gpu_index) {
  PolyBufferMemoryDescriptors rtn;
  // accessing GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    std::lock_guard<std::mutex> render_lock(insitu_poly_mutex_);
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    if (buffers.verts) {
      rtn.vbo_descriptor = buffers.verts->getBufferMemoryDescriptor();
    }
    if (buffers.per_row_data) {
      rtn.ssbo_descriptor = buffers.per_row_data->getBufferMemoryDescriptor();
    }
    if (buffers.line_draw_struct) {
      rtn.line_indirect_vbo_descriptor =
          buffers.line_draw_struct->getBufferMemoryDescriptor();
    }
    if (buffers.poly_draw_struct) {
      rtn.poly_indirect_vbo_descriptor =
          buffers.poly_draw_struct->getBufferMemoryDescriptor();
    }
    if (buffers.poly_rowids) {
      rtn.poly_rowids_ssbo_descriptor = buffers.poly_rowids->getBufferMemoryDescriptor();
    }
  });
  return rtn;
}

void PolyMgr::setPolyTableInSituBuffersPolyDrawBatchInfo(
    const std::string& poly_table_name,
    const size_t gpu_index,
    PolyDrawBatchInfoUqPtr&& poly_draw_batch_info) {
  std::lock_guard<std::mutex> render_lock(insitu_poly_mutex_);
  set_the_poly_table_poly_draw_batch_info(poly_table_name,
                                          std::move(poly_draw_batch_info),
                                          global_context_.getGpuDataFromIndex(gpu_index));
}

void PolyMgr::releasePolyTableInSituBuffersForRendering(
    const PolyBufferPtrs& buffers,
    const size_t gpu_index,
    const QueryDataLayoutShPtr& vert_layout,
    const QueryDataLayoutShPtr& ssbo_layout) {
  // accessing GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    std::lock_guard<std::mutex> render_lock(insitu_poly_mutex_);
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    set_the_poly_table_buffer_ready_for_render(buffers, vert_layout, ssbo_layout);
  });
}

}  // namespace QueryRenderer
