/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>

#include <boost/noncopyable.hpp>

#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/Types.h"

class ResultSet;

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace QueryRenderer {

struct PolyDataPtrs;
struct PolyBufferMemoryDescriptors;
struct PolyBufferPtrs;
struct PolyTableByteData;
struct PolyTableDataInfo;
struct PolyTableLayoutInfo;
class GlobalRenderContext;
class RenderCmdQueue;

class PolyMgr : boost::noncopyable {
 public:
  explicit PolyMgr(GlobalRenderContext& global_context,
                   RenderCmdQueue& command_queue,
                   CudaMgr_Namespace::CudaMgr* cuda_mgr);
  PolyMgr() = delete;

  //
  // in-situ polys
  //

  // create in-situ poly buffers
  // @TODO does this still need the table name for per-table buffer pooling?
  PolyBufferPtrs createPolyTableInSituBuffers(const std::string& poly_table_name,
                                              const size_t gpu_index,
                                              const PolyTableByteData& init_table_data);

  // switch buffers to CUDA mode for Thrust population
  PolyBufferMemoryDescriptors getPolyTableInSituBufferDescriptors(
      const PolyBufferPtrs& buffers,
      const size_t gpu_index);

  // data for poly draw that is added after buffer creation
  void setPolyTableInSituBuffersPolyDrawBatchInfo(
      const std::string& poly_table_name,
      const size_t gpu_index,
      PolyDrawBatchInfoUqPtr&& poly_draw_batch_info);

  // release in-situ poly buffers for rendering (INCOMPLETE)
  void releasePolyTableInSituBuffersForRendering(const PolyBufferPtrs& buffers,
                                                 const size_t gpu_index,
                                                 const QueryDataLayoutShPtr& vert_layout,
                                                 const QueryDataLayoutShPtr& ssbo_layout);

 private:
  mutable std::mutex insitu_poly_mutex_;
  GlobalRenderContext& global_context_;
  RenderCmdQueue& command_queue_;
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
};

}  // namespace QueryRenderer
