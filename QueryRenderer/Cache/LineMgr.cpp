/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Cache/LineMgr.h"

#include "CudaMgr/CudaMgr.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/RenderCmdQueue.h"

namespace QueryRenderer {

namespace {

LineBufferPtrs create_the_line_table_buffers(const std::string& data_table_name,
                                             const LineTableByteData& table_byte_data,
                                             RootPerGpuData& gpu_data) {
  LineBufferPtrs rtn;
  auto& tmp_line_buffer_wk_ptrs =
      gpu_data.createTmpLineBuffersForDataTable(data_table_name);
  if (table_byte_data.num_vertex_bytes > 0) {
    tmp_line_buffer_wk_ptrs.verts =
        gpu_data.getVboBufferPool().getInactiveRsrc(table_byte_data.num_vertex_bytes);
    rtn.verts = tmp_line_buffer_wk_ptrs.verts.lock();
    CHECK(rtn.verts);
  }

  if (table_byte_data.num_index_buffer_bytes > 0) {
    tmp_line_buffer_wk_ptrs.indices = gpu_data.getIboBufferPool().getInactiveRsrc(
        table_byte_data.num_index_buffer_bytes);
    rtn.indices = tmp_line_buffer_wk_ptrs.indices.lock();
    CHECK(rtn.indices);
  }

  if (table_byte_data.num_ssbo_bytes > 0) {
    tmp_line_buffer_wk_ptrs.per_row_data =
        gpu_data.getSsboBufferPool().getInactiveRsrc(table_byte_data.num_ssbo_bytes);
    rtn.per_row_data = tmp_line_buffer_wk_ptrs.per_row_data.lock();
    CHECK(rtn.per_row_data);
  }

  if (table_byte_data.num_indirect_vertex_bytes > 0) {
    tmp_line_buffer_wk_ptrs.indirect_vertex_struct =
        gpu_data.getIndVboBufferPool().getInactiveRsrc(
            table_byte_data.num_indirect_vertex_bytes);
    rtn.indirect_vertex_struct = tmp_line_buffer_wk_ptrs.indirect_vertex_struct.lock();
    CHECK(rtn.indirect_vertex_struct);
  }

  if (table_byte_data.num_indirect_index_bytes > 0) {
    tmp_line_buffer_wk_ptrs.indirect_index_struct =
        gpu_data.getIndIboBufferPool().getInactiveRsrc(
            table_byte_data.num_indirect_index_bytes);
    rtn.indirect_index_struct = tmp_line_buffer_wk_ptrs.indirect_index_struct.lock();
    CHECK(rtn.indirect_index_struct);
  }

  return rtn;
}

void set_the_line_table_buffers_ready_for_render(
    const LineBufferPtrs& line_buffers,
    const QueryDataLayoutShPtr& vert_layout,
    const QueryDataLayoutShPtr& ssbo_layout) {
  // NOTE: it is assumed that all bytes allocated are used
  if (line_buffers.verts && vert_layout) {
    line_buffers.verts->setQueryDataLayout(
        vert_layout, line_buffers.verts->getNumBytes(), 0);
  }

  if (line_buffers.per_row_data && ssbo_layout) {
    line_buffers.per_row_data->setQueryDataLayout(
        ssbo_layout, line_buffers.per_row_data->getNumBytes(), 0);
  }
}

inline void internal_memcpy(void* dest, const void* src, const uint64_t size) {
#ifdef HAVE_CUDA
  cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(
#else
  std::memcpy(reinterpret_cast<int8_t*>(
#endif  // HAVE_CUDA
                   dest),
               src,
               size);
}

inline void internal_set_cuda_context(CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                      const size_t gpu_index) {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr);
  cuda_mgr->setContext(gpu_index);
#endif  // HAVE_CUDA
}

}  // namespace

LineMgr::LineMgr(GlobalRenderContext& global_context,
                 RenderCmdQueue& command_queue,
                 CudaMgr_Namespace::CudaMgr* cuda_mgr,
                 std::mutex& buffer_mutex)
    : global_context_(global_context)
    , command_queue_(command_queue)
    , cuda_mgr_(cuda_mgr)
    , buffer_mutex_(buffer_mutex) {}

void LineMgr::bufferLineData(
    const std::string& vega_data_table_name,
    const std::vector<SqlQueryLineFormatJson::LineDrawBufferData>&
        line_draw_buffer_data_vec,
    const QueryDataLayout::LayoutType& vertex_layout,
    const std::vector<char>& render_query_result_data,
    const LineTableByteData& line_byte_data,
    const std::vector<gfx::IndirectDrawVertexData>& indir_draw_vbo_data,
    const std::vector<gfx::IndirectDrawIndexData>& indir_draw_ibo_data,
    const QueryDataLayoutShPtr& ssbo_layout,
    const QueryDataLayoutShPtr& vert_layout,
    const bool use_index_buffer,
    const size_t bytes_per_column_type,
    const size_t gpu_index) {
  auto& gpu_data = global_context_.getGpuDataFromIndex(gpu_index);

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(buffer_mutex_);

  command_queue_.submit([&] {
    internal_set_cuda_context(cuda_mgr_, gpu_index);

    auto line_data_buffers =
        create_the_line_table_buffers(vega_data_table_name, line_byte_data, gpu_data);

    if (line_data_buffers.verts) {
      auto verts_descriptor = line_data_buffers.verts->getBufferMemoryDescriptor();
      size_t cu_vertex_byte_offset = 0;
      for (const auto& line_draw_buffer_data : line_draw_buffer_data_vec) {
        internal_memcpy(
            verts_descriptor.handle + cu_vertex_byte_offset,
            line_draw_buffer_data.primary_vertices_ptr->data(),
            line_draw_buffer_data.primary_vertices_ptr->size() * bytes_per_column_type);
        cu_vertex_byte_offset +=
            line_draw_buffer_data.primary_vertices_ptr->size() * bytes_per_column_type;
      }
      if (vertex_layout == QueryDataLayout::LayoutType::kVertexSequential) {
        for (const auto& line_draw_buffer_data : line_draw_buffer_data_vec) {
          internal_memcpy(verts_descriptor.handle + cu_vertex_byte_offset,
                          line_draw_buffer_data.secondary_vertices_ptr->data(),
                          line_draw_buffer_data.secondary_vertices_ptr->size() *
                              bytes_per_column_type);
          cu_vertex_byte_offset += line_draw_buffer_data.secondary_vertices_ptr->size() *
                                   bytes_per_column_type;
        }
      }
    }

    if (use_index_buffer && line_byte_data.num_index_buffer_bytes > 0) {
      CHECK(line_data_buffers.indices);
      auto indices_descriptor = line_data_buffers.indices->getBufferMemoryDescriptor();

      size_t cu_index_byte_offset = 0;
      for (const auto& line_draw_buffer_data : line_draw_buffer_data_vec) {
        internal_memcpy(indices_descriptor.handle,
                        line_draw_buffer_data.indices_ptr->data() + cu_index_byte_offset,
                        line_draw_buffer_data.indices_ptr->size() * sizeof(unsigned int));
        cu_index_byte_offset +=
            line_draw_buffer_data.indices_ptr->size() * sizeof(unsigned int);
      }
    }

    if (line_data_buffers.per_row_data) {
      auto per_row_data_descriptor =
          line_data_buffers.per_row_data->getBufferMemoryDescriptor();
      internal_memcpy(per_row_data_descriptor.handle,
                      render_query_result_data.data(),
                      line_byte_data.num_ssbo_bytes);
    }

    if (line_data_buffers.indirect_vertex_struct) {
      auto indirect_vertex_struct_descriptor =
          line_data_buffers.indirect_vertex_struct->getBufferMemoryDescriptor();
      internal_memcpy(indirect_vertex_struct_descriptor.handle,
                      indir_draw_vbo_data.data(),
                      line_byte_data.num_indirect_vertex_bytes);
    }

    if (use_index_buffer && line_data_buffers.indirect_index_struct) {
      auto indirect_index_struct_descriptor =
          line_data_buffers.indirect_index_struct->getBufferMemoryDescriptor();
      internal_memcpy(indirect_index_struct_descriptor.handle,
                      indir_draw_ibo_data.data(),
                      line_byte_data.num_indirect_index_bytes);
    }
    set_the_line_table_buffers_ready_for_render(
        line_data_buffers, vert_layout, ssbo_layout);
  });
}

//
// in-situ lines
//

LineBufferPtrs LineMgr::createLineTableInSituBuffers(
    const std::string& line_table_name,
    const size_t gpu_index,
    const LineTableByteData& init_table_data) {
  std::lock_guard<std::mutex> render_lock(line_mutex_);

  auto& gpu_data = global_context_.getGpuDataFromIndex(gpu_index);

  LineBufferPtrs rtn;
  // creating GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    rtn = create_the_line_table_buffers(line_table_name, init_table_data, gpu_data);
  });
  return rtn;
}

LineBufferMemoryDescriptors LineMgr::getLineTableInSituBufferDescriptors(
    const LineBufferPtrs& line_buffers,
    const size_t gpu_index) {
  std::lock_guard<std::mutex> render_lock(line_mutex_);

  LineBufferMemoryDescriptors rtn;
  // accessing GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    if (line_buffers.verts) {
      rtn.vbo_descriptor = line_buffers.verts->getBufferMemoryDescriptor();
    }
    if (line_buffers.per_row_data) {
      rtn.ssbo_descriptor = line_buffers.per_row_data->getBufferMemoryDescriptor();
    }
    if (line_buffers.indirect_vertex_struct) {
      rtn.line_indirect_vbo_descriptor =
          line_buffers.indirect_vertex_struct->getBufferMemoryDescriptor();
    }
  });
  return rtn;
}

void LineMgr::releaseLineTableInSituBuffersForRendering(
    const LineBufferPtrs& line_buffers,
    const size_t gpu_index,
    const QueryDataLayoutShPtr& vert_layout,
    const QueryDataLayoutShPtr& ssbo_layout) {
  std::lock_guard<std::mutex> render_lock(line_mutex_);

  // accessing GPU resources, so must run on the render thread
  command_queue_.submit([&] {
    internal_set_cuda_context(cuda_mgr_, gpu_index);
    set_the_line_table_buffers_ready_for_render(line_buffers, vert_layout, ssbo_layout);
  });
}

}  // namespace QueryRenderer
