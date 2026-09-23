/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/QueryBufferManager.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/DataMgr.h"
#include "GfxDriver/DeviceContext.h"
#include "QueryRenderer/QueryBufferAllocator.h"

#define LOG_QUERY_BUFFER_LIFETIME false

namespace QueryRenderer {

QueryBufferManager::QueryBufferManager(const gfx::DeviceContext& device_ctx,
                                       Data_Namespace::DataMgr* data_mgr,
                                       const bool renderer_enable_slab_allocation)
    : device_ctx_(device_ctx)
    , data_mgr_{data_mgr}
    , renderer_enable_slab_allocation_{renderer_enable_slab_allocation}
    , slab_address_table_{std::make_unique<SlabAddressTable>(
          device_ctx.getResourceManager())}
    , slabs_changed_{false}
    , slabs_changed_cbid_{0u} {
#ifdef HAVE_CUDA
  if (data_mgr_ && data_mgr_->getCudaMgr()) {
    // register interest in device memory allocation map changed
    slabs_changed_cbid_ =
        data_mgr_->getCudaMgr()->getDeviceMemoryAllocationMap().registerMapChangedCB(
            [this](const heavyai::UUID gpu_uuid, const bool is_slab) {
              this->slabsChangedCB(gpu_uuid, is_slab);
            });

    // create buffer allocator
    buffer_allocator_ = std::make_unique<QueryBufferAllocator>(
        device_ctx, *this, data_mgr_, device_ctx_.getGpuId());
  }
#endif
}

QueryBufferManager::~QueryBufferManager() {
#ifdef HAVE_CUDA
  if (data_mgr_ && data_mgr_->getCudaMgr()) {
    // unregister interest in device memory allocation map changed
    if (slabs_changed_cbid_) {
      data_mgr_->getCudaMgr()->getDeviceMemoryAllocationMap().unregisterMapChangedCB(
          slabs_changed_cbid_);
    }

    // destroy buffer allocator
    buffer_allocator_ = nullptr;
  }
#endif
}

QueryBufferManager::BuffersAndDescriptor QueryBufferManager::createQueryBuffer(
    std::string_view resource_tracking_string,
    const gfx::BufferCreateInfo& create_info) {
  BuffersAndDescriptor result;

  auto& resource_mgr = device_ctx_.getResourceManager();

  // if CUDA available, allocate memory with CUDA and wrap a device-local buffer
#ifdef HAVE_CUDA
  if (data_mgr_ && data_mgr_->getCudaMgr()) {
    if (renderer_enable_slab_allocation_) {
      // buffer (includes allocation)
      result.buffer = resource_mgr.createBuffer(
          resource_tracking_string, create_info, buffer_allocator_);

      // actual CUDA address is slab plus offset
      result.descriptor.handle =
          reinterpret_cast<int8_t*>(result.buffer->getAllocationBasePtr() +
                                    result.buffer->getAllocationOffsetBytes());
    } else {
      // get the CUDA device_num which may not be the same as our gpu_id
      // find the index of the DeviceProperties with the matching UUID
      // this is a bit ugly, but this code will go away with slab-allocation
      int cuda_device_num = 0;
      auto const& all_cuda_device_properties =
          data_mgr_->getCudaMgr()->getAllDeviceProperties();
      auto const num_cuda_devices = static_cast<int>(all_cuda_device_properties.size());
      for (; cuda_device_num < num_cuda_devices; cuda_device_num++) {
        if (all_cuda_device_properties[cuda_device_num].uuid ==
            device_ctx_.getGpuUUID()) {
          break;
        }
      }
      CHECK_LT(cuda_device_num, num_cuda_devices);

      // allocate memory
      try {
        result.descriptor.handle =
            data_mgr_->getCudaMgr()->allocateDeviceMem(create_info.size, cuda_device_num);
      } catch (CudaMgr_Namespace::CudaErrorException& e) {
        if (e.getStatus() == CUDA_ERROR_OUT_OF_MEMORY) {
          // details to error log
          LOG(ERROR) << "Failed to allocate " << create_info.size << " bytes on GPU "
                     << cuda_device_num << " for " << resource_tracking_string;
          // throw Renderer exception to be caught by RenderHandler
          throw gfx::OutOfGpuMemoryError("Failed to allocate GPU memory for " +
                                         std::string(resource_tracking_string));
        }
        // otherwise re-throw CUDA exception
        std::rethrow_exception(std::current_exception());
      }
      CHECK(result.descriptor.handle);

      // get map and find the allocation we just made
      auto const device_memory_allocation_map =
          data_mgr_->getCudaMgr()->getDeviceMemoryAllocationMap().getMap();
      auto const itr = device_memory_allocation_map.find(
          reinterpret_cast<CudaMgr_Namespace::DeviceMemoryAllocationMap::DevicePtr>(
              result.descriptor.handle));
      CHECK(itr != device_memory_allocation_map.end())
          << "Failed to find CUDA allocation";

      // the actual CreateInfo
      gfx::BufferCreateInfo ci{create_info};

      // get handle and export FD
      ci.import_allocation_fd = data_mgr_->getCudaMgr()->exportHandle(itr->second.handle);
      CHECK_GE(ci.import_allocation_fd, 0) << "Failed to export CUDA allocation";

      // for an imported CUDA buffer, buffer must be DeviceLocal
      // previously this would have been ExternalApi, but that's
      // only when you then want to EXPORT the buffer back to CUDA
      // which we are no longer doing
      ci.access_type = gfx::BufferAccessType::kDeviceLocal;

      // create buffer
      result.buffer = resource_mgr.createBuffer(resource_tracking_string, ci);
      CHECK(result.buffer);
    }
  }
#endif

  // otherwise we need a host-visible buffer
  if (result.descriptor.handle == nullptr) {
    // the actual CreateInfo
    gfx::HostVisibleBufferCreateInfo ci{
        create_info.buffer_type, create_info.size, create_info.usage};

    // create buffer
    result.host_visible_buffer =
        resource_mgr.createHostVisibleBuffer(resource_tracking_string, ci);
    CHECK(result.host_visible_buffer);

    // map it (permanently)
    result.host_visible_buffer->map((void**)&result.descriptor.handle);
    CHECK(result.descriptor.handle);
  }

  // the size
  result.descriptor.num_bytes = create_info.size;

  if (LOG_QUERY_BUFFER_LIFETIME) {
    std::cout << "**** Creating QueryBuffer '" << result.buffer->getTrackingData().origin
              << "'" << std::endl;
    std::cout << "****   cuda address   0x" << std::hex
              << reinterpret_cast<uint64_t>(result.descriptor.handle) << std::dec
              << std::endl;
    std::cout << "****   vulkan address 0x" << std::hex
              << result.buffer->getDeviceAddress() << std::dec << std::endl;
    std::cout << "****   num_bytes      0x" << std::hex << result.descriptor.num_bytes
              << std::dec << " (" << result.descriptor.num_bytes << ")" << std::endl;
    std::cout << "****   offset_bytes   0x" << std::hex
              << result.buffer->getAllocationOffsetBytes() << std::dec << " ("
              << result.buffer->getAllocationOffsetBytes() << ")" << std::endl;
  }

  // done
  return result;
}

void QueryBufferManager::destroyQueryBuffer(
    QueryBufferManager::BuffersAndDescriptor&& buffers_and_descriptor) {
  if (LOG_QUERY_BUFFER_LIFETIME) {
    std::cout << "**** Destroying QueryBuffer '"
              << buffers_and_descriptor.buffer->getTrackingData().origin << "'"
              << std::endl;
    std::cout << "****   cuda address   0x" << std::hex
              << reinterpret_cast<uint64_t>(buffers_and_descriptor.descriptor.handle)
              << std::dec << std::endl;
    std::cout << "****   vulkan address 0x" << std::hex
              << buffers_and_descriptor.buffer->getDeviceAddress() << std::dec
              << std::endl;
    std::cout << "****   num_bytes      0x" << std::hex
              << buffers_and_descriptor.descriptor.num_bytes << std::dec << " ("
              << buffers_and_descriptor.descriptor.num_bytes << ")" << std::endl;
    std::cout << "****   offset_bytes   0x" << std::hex
              << buffers_and_descriptor.buffer->getAllocationOffsetBytes() << std::dec
              << " (" << buffers_and_descriptor.buffer->getAllocationOffsetBytes() << ")"
              << std::endl;
  }

  // destroy buffer
  auto& resource_mgr = device_ctx_.getResourceManager();
  if (buffers_and_descriptor.buffer) {
    // destroy buffer
    CHECK(!buffers_and_descriptor.host_visible_buffer);
    resource_mgr.destroyBuffer(std::move(buffers_and_descriptor.buffer));

#ifdef HAVE_CUDA
    // free allocation
    if (data_mgr_ && data_mgr_->getCudaMgr() && !renderer_enable_slab_allocation_) {
      CHECK(buffers_and_descriptor.descriptor.handle);
      CHECK_GT(buffers_and_descriptor.descriptor.num_bytes, 0u);
      data_mgr_->getCudaMgr()->freeDeviceMem(buffers_and_descriptor.descriptor.handle);
    }
#endif
  } else if (buffers_and_descriptor.host_visible_buffer) {
    // unmap and destroy host-visible buffer
    CHECK(buffers_and_descriptor.host_visible_buffer->isMapped());
    buffers_and_descriptor.host_visible_buffer->unmap();
    resource_mgr.destroyHostVisibleBuffer(
        std::move(buffers_and_descriptor.host_visible_buffer));
  }
}

void QueryBufferManager::destroyResources() {
  slab_address_table_->reset();
  destroyWrapperBuffers();
}

const gfx::BufferWrapper& QueryBufferManager::getSlabAddressTableBuffer() const {
  CHECK(slab_address_table_);
  return slab_address_table_->getTableBuffer();
}

void QueryBufferManager::updateSlabAddressTableAndBuffers() {
#ifdef HAVE_CUDA
  if (!slabsChanged()) {
    return;
  }

  // tolerate lack of DataMgr/CudaMgr
  if (!data_mgr_ || !data_mgr_->getCudaMgr()) {
    return;
  }

  // get all the allocations (all GPUs)
  auto const device_memory_allocation_map =
      data_mgr_->getCudaMgr()->getDeviceMemoryAllocationMap().getMap();

  auto& resource_mgr = device_ctx_.getResourceManager();

  // iterate all the allocations
  // should be in ascending cuda_addr order
  for (auto const& [cuda_addr, allocation] : device_memory_allocation_map) {
    // there will be slab and non-slab allocations in the map
    // we only care about slab allocations, and only ones on this GPU
    // only slab allocation changes will trigger this anyway
    if (allocation.device_uuid == device_ctx_.getGpuUUID() && allocation.is_slab) {
      // do we already have a wrapper for this slab?
      // if we do not, create one
      auto itr = wrapper_buffers_.find(cuda_addr);
      if (itr == wrapper_buffers_.end()) {
        // export this allocation
        int exported_fd = data_mgr_->getCudaMgr()->exportHandle(allocation.handle);
        CHECK(exported_fd);

        // create the wrapper buffer
        gfx::BufferWrapperUqPtr wrapper_buffer;
        try {
          // construct meaningful name
          std::stringstream ss;
          ss << "Slab " << slab_address_table_->numEntries() << " Wrapper Buffer, GPU "
             << allocation.device_uuid << ", Device Address 0x" << std::hex << cuda_addr
             << std::dec << ", Size " << allocation.size;

          // create buffer
          wrapper_buffer =
              resource_mgr.createBuffer(ss.str(),
                                        {gfx::BufferType::kSlabWrapperBuffer,
                                         allocation.size,
                                         gfx::BufferUsageBits::kStorageBufferBit |
                                             gfx::BufferUsageBits::kDeviceAddressBit,
                                         gfx::BufferAccessType::kDeviceLocal,
                                         exported_fd});
        } catch (std::exception&) {
          // close the file descriptor before throwing further
          close(exported_fd);

          // reset and destroy all buffers
          slab_address_table_->reset();
          destroyWrapperBuffers();

          // and then...
          throw;
        }

        // store buffer
        CHECK(wrapper_buffers_.try_emplace(cuda_addr, std::move(wrapper_buffer)).second);
      }
    }
  }

  // now rebuild the slab address table to match
  // the wrapper buffers map will be in slab base address order
  slab_address_table_->reset();
  for (auto const& itr : wrapper_buffers_) {
    auto const cuda_addr = itr.first;
    auto const vulkan_addr = itr.second->getDeviceAddress();
    slab_address_table_->addEntry(cuda_addr, vulkan_addr);
  }
  slab_address_table_->finalizeAndUpdateBuffer();
#endif
}

void QueryBufferManager::destroyWrapperBuffers() {
  auto& resource_mgr = device_ctx_.getResourceManager();

  // destroy all buffers
  for (auto& itr : wrapper_buffers_) {
    resource_mgr.destroyBuffer(std::move(itr.second));
  }
  wrapper_buffers_.clear();
}

void QueryBufferManager::slabsChangedCB(const heavyai::UUID gpu_uuid,
                                        const bool is_slab) {
  std::lock_guard<std::mutex> lock(slabs_changed_mutex_);
  if (gpu_uuid == device_ctx_.getGpuUUID() && is_slab) {
    // slabs on this GPU have changed
    slabs_changed_ = true;
  }
}

bool QueryBufferManager::slabsChanged() {
  std::lock_guard<std::mutex> lock(slabs_changed_mutex_);
  // clear and return whether it was set
  bool changed = slabs_changed_;
  slabs_changed_ = false;
  return changed;
}

gfx::BufferAllocatorShPtr QueryBufferManager::getBufferAllocator() const {
  return buffer_allocator_;
}

}  // namespace QueryRenderer
