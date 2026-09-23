/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

#include "DataMgr/DataMgr.h"
#include "GfxDriver/Resources/BufferAllocator.h"
#include "GfxInterop/BufferMemoryDescriptor.h"
#include "QueryRenderer/QueryBufferAllocator.h"
#include "QueryRenderer/SlabAddressTable.h"
#include "Shared/uuid.h"

namespace QueryRenderer {

class QueryBufferManager : boost::noncopyable {
 public:
  explicit QueryBufferManager(const gfx::DeviceContext& device_ctx,
                              Data_Namespace::DataMgr* data_mgr,
                              const bool renderer_enable_slab_allocation);
  QueryBufferManager() = delete;
  ~QueryBufferManager();

  struct BuffersAndDescriptor {
    gfx::BufferWrapperUqPtr buffer;
    gfx::HostVisibleBufferWrapperUqPtr host_visible_buffer;
    gfx::BufferMemoryDescriptor descriptor;
  };

  BuffersAndDescriptor createQueryBuffer(std::string_view resource_tracking_string,
                                         const gfx::BufferCreateInfo& create_info);

  void destroyQueryBuffer(BuffersAndDescriptor&& buffer_and_desc);

  void destroyResources();

  const gfx::BufferWrapper& getSlabAddressTableBuffer() const;
  void updateSlabAddressTableAndBuffers();

  void slabsChangedCB(const heavyai::UUID gpu_uuid, const bool is_slab);

  gfx::BufferAllocatorShPtr getBufferAllocator() const;

 private:
  const gfx::DeviceContext& device_ctx_;
  Data_Namespace::DataMgr* data_mgr_;
  const bool renderer_enable_slab_allocation_;

  std::unique_ptr<SlabAddressTable> slab_address_table_;
  std::map<uint64_t, gfx::BufferWrapperUqPtr> wrapper_buffers_;

  std::mutex slabs_changed_mutex_;
  bool slabs_changed_;
  uint32_t slabs_changed_cbid_;

  gfx::BufferAllocatorShPtr buffer_allocator_;

  void destroyWrapperBuffers();
  bool slabsChanged();

  friend class QueryBufferAllocator;
};

}  // namespace QueryRenderer
