/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/Interop/InteropBufferInfo.h"
#include "QueryRenderer/Interop/LayoutAttrInfo.h"
#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/Types.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace QueryRenderer {

class InteropBufferMgr : ::boost::noncopyable {
 public:
  struct InteropBufferAttrInfo : public LayoutAttrInfo {
    using InteropBufferInfoMap = std::map<GpuId, InteropBufferInfo>;
    InteropBufferInfoMap buffer_data_map;

    InteropBufferAttrInfo(const std::string& attr_name,
                          const SQLTypeInfo& type_info,
                          const gfx::BufferLayoutShPtr& buffer_layout)
        : LayoutAttrInfo(attr_name, type_info, buffer_layout) {}
  };

  using InteropBufferAttrSet = ::boost::multi_index_container<
      InteropBufferAttrInfo,
      ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
          ::boost::multi_index::
              member<LayoutAttrInfo, const std::string, &LayoutAttrInfo::attr_name>>>>;

  InteropBufferMgr(const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                   const GlobalRenderContext& global_context);
  ~InteropBufferMgr();

  InteropBufferAttrSet getThrustBuffersForAttrs(
      const BaseDataTableShPtr& source_data_table,
      const LayoutAttrInfoSet& attr_infos);

 private:
#ifdef HAVE_CUDA
  CUcontext curr_cuda_ctx_;
  const CudaMgr_Namespace::CudaMgr* cuda_mgr_;
#endif  // HAVE_CUDA
  const GlobalRenderContext& global_context_;

  struct InteropBufferAttrSetModifier {
    InteropBufferAttrSetModifier(const size_t gpu_idx,
                                 const gfx::BufferMemoryDescriptor& interop_descriptor,
                                 const gfx::BufferWrapper* layout_buffer,
                                 const int64_t invalid_key)
        : gpu_idx_(gpu_idx)
        , buffer_info_({interop_descriptor, layout_buffer, invalid_key}) {}

    void operator()(InteropBufferAttrInfo& buffer_info) {
      CHECK(buffer_info.buffer_data_map.find(gpu_idx_) ==
            buffer_info.buffer_data_map.end());
      buffer_info.buffer_data_map.insert({gpu_idx_, buffer_info_});
    }

   private:
    const size_t gpu_idx_;
    const InteropBufferInfo buffer_info_;
  };

  struct InitialLayoutBufferState {
    bool initially_mapped = false;
    BufferMemoryDescriptor mapped_buffer_descriptor = {};
  };
  using LayoutInteropHandleMap =
      std::unordered_map<QueryLayoutBufferShPtr, InitialLayoutBufferState>;
  using InteropHandleMap = std::map<size_t, LayoutInteropHandleMap>;
  InteropHandleMap mapped_buffers_;
};

}  // namespace QueryRenderer
