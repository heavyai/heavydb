/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Interop/InteropBufferMgr.h"

#include "CudaMgr/CudaMgr.h"
#include "GfxInterop/Utils/CudaErrorCheck.h"
#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "Shared/scope.h"

namespace QueryRenderer {

QueryDataType LayoutAttrInfo::getTypeFromLayoutAttr(const LayoutAttrInfo& attr_info) {
  CHECK(attr_info.buffer_layout);
  const auto& layout_attr_info =
      attr_info.buffer_layout->getAttributeInfo(attr_info.attr_name);
  return convertToQueryDataType(layout_attr_info.type);
}

InteropBufferMgr::InteropBufferMgr(const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                   const GlobalRenderContext& global_context)
    :
#ifdef HAVE_CUDA
    cuda_mgr_(cuda_mgr)
    ,
#endif
    global_context_(global_context) {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr_);
  CHECK_RENDER_CUDA_ERRORS(cuCtxGetCurrent(&curr_cuda_ctx_), 0);
#endif  // HAVE_CUDA
}

InteropBufferMgr::~InteropBufferMgr() {
#ifdef HAVE_CUDA
  ScopeGuard atexit = [this] {
    CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(curr_cuda_ctx_), 0);
  };
  auto& cuda_ctx_vector = cuda_mgr_->getDeviceContexts();
#endif  // HAVE_CUDA

  for (auto& [gpu_idx, layout_descriptor_map] : mapped_buffers_) {
#ifdef HAVE_CUDA
    CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cuda_ctx_vector[gpu_idx]), gpu_idx);
#endif
    for (auto& [buffer_layout, initial_buffer_state] : layout_descriptor_map) {
      // NOTE: if the initial buffer state was unmapped, means the buffer was mapped
      // for this specific interop use, therefore unmap it at close. Otherwise let
      // something else handle the unmapping
      if (buffer_layout && !initial_buffer_state.initially_mapped &&
          initial_buffer_state.mapped_buffer_descriptor.handle) {
        buffer_layout->unmap();
      }
    }
  }
}

InteropBufferMgr::InteropBufferAttrSet InteropBufferMgr::getThrustBuffersForAttrs(
    const BaseDataTableShPtr& source_data_ptr,
    const LayoutAttrInfoSet& attr_infos) {
  InteropBufferAttrSet buffer_inputs;
  std::for_each(
      attr_infos.begin(),
      attr_infos.end(),
      [this, &buffer_inputs, &source_data_ptr](const auto& attr_info) {
        auto buffer_map = source_data_ptr->getAttributeDataBuffers(attr_info.attr_name);
        if (buffer_map.size()) {
          for (auto& [gpu_id, buffer_layout_wk_ptr] : buffer_map) {
            auto buffer_layout = buffer_layout_wk_ptr.lock();
            CHECK(buffer_layout);

            // We're only grabbing the query layout buffer here to get at the
            // value of the invalid key.
            // If invalid rows are purged before rendering, this can be removed
            // because we'd be assured that no invalid keys exist.
            // Or if we can settle on a static value for the invalid key and ensure
            // that "key" is used as an internal keyword and can't be used elsewhere,
            // than this can also be removed.
            auto invalid_key{std::numeric_limits<int64_t>::max()};
            auto* query_layout_buffer = buffer_layout.get();
            if (query_layout_buffer) {
              auto query_data_layout = query_layout_buffer->getQueryDataLayout();
              if (query_data_layout) {
                invalid_key = query_data_layout->getInvalidKey();
              }
            }

            const auto start_gpu_id = global_context_.getStartGpuId();
            auto gpu_idx = gpu_id - start_gpu_id;

#ifdef HAVE_CUDA
            auto& cuda_ctx_vector = cuda_mgr_->getDeviceContexts();
            auto cuda_ctx = cuda_ctx_vector[gpu_idx];
            CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cuda_ctx), gpu_id);
#endif  // HAVE_CUDA

            auto& layout_to_interop_handle_map =
                mapped_buffers_.try_emplace(gpu_idx, LayoutInteropHandleMap())
                    .first->second;

            auto layout_handle_map_itr = layout_to_interop_handle_map.find(buffer_layout);
            if (layout_handle_map_itr == layout_to_interop_handle_map.end()) {
              auto& qrm_per_gpu_data = global_context_.getRootPerGpuData();
              auto per_gpu_itr = qrm_per_gpu_data.find(gpu_id);
              CHECK(per_gpu_itr != qrm_per_gpu_data.end());
              auto const initially_mapped = buffer_layout->isMapped();
              auto interop_handle = buffer_layout->getBufferMemoryDescriptor();
              auto insert = layout_to_interop_handle_map.insert(
                  {buffer_layout, {initially_mapped, interop_handle}});
              CHECK(insert.second);
              layout_handle_map_itr = insert.first;
            }

            auto const* layout_buffer = layout_handle_map_itr->first->getBufferWrapper();
            CHECK(layout_buffer);

            // NOTE: can't use try_emplace here because buffer_inputs is currently a
            // boost::mutli_index_container. If that's changed to a std container,
            // try_emplace could be used instead
            auto buffer_inputs_itr = buffer_inputs.find(attr_info.attr_name);
            if (buffer_inputs_itr == buffer_inputs.end()) {
              auto insert = buffer_inputs.emplace(
                  attr_info.attr_name, attr_info.type_info, attr_info.buffer_layout);
              CHECK(insert.second);
              buffer_inputs_itr = insert.first;
            }

            buffer_inputs.modify(
                buffer_inputs_itr,
                InteropBufferAttrSetModifier(
                    gpu_idx,
                    layout_handle_map_itr->second.mapped_buffer_descriptor,
                    layout_buffer,
                    invalid_key));
          }
        }
      });
  return buffer_inputs;
}

}  // namespace QueryRenderer
