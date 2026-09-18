/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include <boost/noncopyable.hpp>

#include "QueryRenderer/Data/Parsers/SqlQueryLineFormatJson.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/PerGpuData.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace QueryRenderer {

class RenderCmdQueue;
class GlobalRenderContext;

class LineMgr : boost::noncopyable {
 public:
  explicit LineMgr(GlobalRenderContext& global_context,
                   RenderCmdQueue& command_queue,
                   CudaMgr_Namespace::CudaMgr* cuda_mgr,
                   std::mutex& buffer_mutex);
  LineMgr() = delete;

  //
  // Cached lines
  //
  void bufferLineData(const std::string& vega_data_table_name,
                      const std::vector<SqlQueryLineFormatJson::LineDrawBufferData>&
                          line_draw_buffer_data_vector,
                      const QueryDataLayout::LayoutType& vertex_layout,
                      const std::vector<char>& render_query_result_data,
                      const LineTableByteData& line_byte_data,
                      const std::vector<gfx::IndirectDrawVertexData>& indir_draw_vbo_data,
                      const std::vector<gfx::IndirectDrawIndexData>& indir_draw_ibo_data,
                      const QueryDataLayoutShPtr& ssbo_layout,
                      const QueryDataLayoutShPtr& vert_layout,
                      const bool use_index_buffer,
                      const size_t bytes_per_row,
                      const size_t gpu_index);

  //
  // in-situ lines
  //
  // create in-situ poly buffers
  // @TODO does this still need the table name for per-table buffer pooling?
  LineBufferPtrs createLineTableInSituBuffers(const std::string& line_table_name,
                                              const size_t gpu_index,
                                              const LineTableByteData& init_table_data);

  // switch buffers to CUDA mode for Thrust population
  LineBufferMemoryDescriptors getLineTableInSituBufferDescriptors(
      const LineBufferPtrs& line_buffers,
      const size_t gpu_index);

  // release in-situ line buffers for rendering (INCOMPLETE)
  void releaseLineTableInSituBuffersForRendering(const LineBufferPtrs& line_buffers,
                                                 const size_t gpu_index,
                                                 const QueryDataLayoutShPtr& vert_layout,
                                                 const QueryDataLayoutShPtr& ssbo_layout);

 private:
  GlobalRenderContext& global_context_;
  RenderCmdQueue& command_queue_;
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  std::mutex& buffer_mutex_;
  std::mutex line_mutex_;
};

}  // namespace QueryRenderer
