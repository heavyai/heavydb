/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/QueryBufferAllocator.h"

#include "CudaMgr/CudaMgr.h"
#include "QueryRenderer/QueryBufferManager.h"

namespace QueryRenderer {

QueryBufferAllocator::QueryBufferAllocator(const gfx::DeviceContext& device_ctx,
                                           QueryBufferManager& query_buffer_manager,
                                           Data_Namespace::DataMgr* data_mgr,
                                           const int gpu_id)
    : device_ctx_{device_ctx}
    , data_mgr_{data_mgr}
#ifdef HAVE_CUDA
    , query_buffer_manager_{query_buffer_manager}
    , gpu_id_{gpu_id}
#endif
{
}

gfx::BufferAllocationUqPtr QueryBufferAllocator::alloc(const uint64_t num_bytes) {
  // enabled?
  if (!data_mgr_) {
    return nullptr;
  }

#ifdef HAVE_CUDA
  // allocate memory
  auto* abstract_buffer = data_mgr_->alloc(Data_Namespace::GPU_LEVEL, gpu_id_, num_bytes);
  if (!abstract_buffer) {
    return nullptr;
  }
  auto const allocation_base_ptr =
      reinterpret_cast<uint64_t>(abstract_buffer->getMemoryPtr());

  // get map and find the allocation we just made
  // @TODO(se) this may have to change to work on multi-GPU!
  auto& map = data_mgr_->getCudaMgr()->getDeviceMemoryAllocationMap();
  auto const& [slab_base_ptr, allocation] = map.getAllocation(allocation_base_ptr);
  CHECK_EQ(allocation.device_num, static_cast<int>(gpu_id_));

  // compute the offset
  auto const offset_bytes = allocation_base_ptr - slab_base_ptr;

  // update ourselves as slabs have likely changed
  query_buffer_manager_.updateSlabAddressTableAndBuffers();

  // find the corresponding wrapper buffer via the slab address table
  auto const& slab_address_table = query_buffer_manager_.slab_address_table_;
  auto const& wrapper_buffers = query_buffer_manager_.wrapper_buffers_;
  auto const num_slabs = slab_address_table->numEntries();
  for (size_t i = 0; i < num_slabs; i++) {
    auto const entry = slab_address_table->getEntry(i);
    if (entry.first == slab_base_ptr) {
      // get the buffer
      auto itr = wrapper_buffers.find(slab_base_ptr);
      CHECK(itr != wrapper_buffers.end());
      auto& buffer = itr->second->getBuffer();

      // store abstract buffer in map keyed by allocation base ptr
      CHECK(abstract_buffers_.emplace(allocation_base_ptr, abstract_buffer).second);

      // success
      return std::make_unique<gfx::BufferAllocation>(
          buffer, slab_base_ptr, num_bytes, offset_bytes);
    }
  }

  // failed
  CHECK(false) << "Failed to complete allocation from slab";
#endif
  return nullptr;
}

void QueryBufferAllocator::free(gfx::BufferAllocationUqPtr buffer_allocation) {
  // derive allocation base ptr
  CHECK(buffer_allocation);
  auto const allocation_base_ptr =
      buffer_allocation->base_ptr + buffer_allocation->offset_bytes;
  buffer_allocation = nullptr;

  // find matching abstract buffer
  auto itr = abstract_buffers_.find(allocation_base_ptr);
  CHECK(itr != abstract_buffers_.end()) << "Failed to find abstract buffer";

  // free memory
  auto* abstract_buffer = itr->second;
  data_mgr_->free(abstract_buffer);

  // remove from map
  abstract_buffers_.erase(itr);
}

void QueryBufferAllocator::validateCreateInfo(const gfx::BufferCreateInfo& create_info) {
  CHECK_EQ(create_info.import_allocation_fd, -1)
      << "Invalid import_allocation_fd for slab-allocated index buffer";
  CHECK_EQ(create_info.access_type, gfx::BufferAccessType::kDeviceLocal)
      << "Invalid access_type for slab-allocated index buffer";
  if (create_info.buffer_type == gfx::BufferType::kIndexBuffer) {
    CHECK_EQ(create_info.size, 0ULL)
        << "Invalid (non-zero) size for slab-allocated index buffer";
  }
}

}  // namespace QueryRenderer
