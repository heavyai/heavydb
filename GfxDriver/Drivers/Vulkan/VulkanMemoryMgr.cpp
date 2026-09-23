/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"

#include <fcntl.h>

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanPlatformUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/Utils/LoggingUtils.h"

#define COUT_DEBUG 0
#if COUT_DEBUG
#include <iostream>
#endif

namespace gfx {

VulkanAllocation::VulkanAllocation(const VulkanDeviceContext& device_ctx,
                                   uint64_t size,
                                   VkDeviceMemory handle)
    : VulkanAllocation(AllocationType::kGeneric, device_ctx, size, handle) {}

VulkanAllocation::VulkanAllocation(AllocationType alloc_type,
                                   const VulkanDeviceContext& device_ctx,
                                   uint64_t size,
                                   VkDeviceMemory handle)
    : alloc_type_{alloc_type}, device_ctx_{device_ctx}, size_{size}, handle_{handle} {}

uint64_t VulkanAllocation::size() {
  return size_;
}

std::atomic<uint32_t> VulkanExportableAllocation::total_export_count_ = 0u;

int VulkanExportableAllocation::exportHandle(const std::string& buffer_name,
                                             std::optional<bool> is_dest) {
  int fd = 0;

  VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
  vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  vkMemoryGetFdInfoKHR.pNext = NULL;
  vkMemoryGetFdInfoKHR.memory = handle_;
  vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

  auto vkres = device_ctx_.getFunctions().vkGetMemoryFdKHR(
      device_ctx_.getHandle(), &vkMemoryGetFdInfoKHR, &fd);

  // increment atomic export counts
  auto const ec = ++export_count_;
  auto const tec = ++total_export_count_;

  auto const gpu_id = device_ctx_.getGpuId();
  auto const buffer_usage =
      (is_dest ? (*is_dest ? "Destination" : "Source") : "Unspecified");

  // failure logging (verbose)
  if (vkres != VK_SUCCESS) {
    // log GPU ID and memory dump
    LOG(WARNING) << "**** Vulkan Memory Export failed";
    LOG(WARNING) << "****   VK_ERROR Code   : " << vkres;
    LOG(WARNING) << "****   GPU             : " << gpu_id;
    LOG(WARNING) << "****   Allocation      : " << (void*)this;
    LOG(WARNING) << "****   Buffer Name     : " << buffer_name;
    LOG(WARNING) << "****   Size            : " << size_;
    LOG(WARNING) << "****   FD              : " << fd;
    LOG(WARNING) << "****   Exp Count       : " << ec;
    LOG(WARNING) << "****   Total Exp Count : " << tec;
    LOG(WARNING) << "****   Buffer Usage    : " << buffer_usage;
    LOG(WARNING) << "****   Device Memory Budget";
    std::stringstream ss1;
    device_ctx_.logMemoryBudgetInfo(ss1);
    LOG(WARNING) << ss1.str();
    LOG(WARNING) << "****   Device Resource Memory Summary";
    std::stringstream ss2;
    device_ctx_.getResourceManager().logMemorySummary(ss2);
    LOG(WARNING) << ss2.str();
  }

  CHECK_VKRESULT(vkres, "Failed to export Vulkan memory handle.");

  // success logging (one line)
  VLOG(1) << "Exported Memory Handle for GPU " << gpu_id << ", Allocation " << (void*)this
          << ", Buffer Name '" << buffer_name << "', Size " << size_ << ", FD " << fd
          << ", Exp Count " << ec << ", Total Exp Count " << tec << ", Buffer Usage "
          << buffer_usage;

  return fd;
}

VulkanExportableAllocation::VulkanExportableAllocation(
    const VulkanDeviceContext& device_ctx,
    uint64_t size,
    VkDeviceMemory handle)
    : VulkanAllocation(AllocationType::kExportable, device_ctx, size, handle) {}

VulkanImportedAllocation::VulkanImportedAllocation(const VulkanDeviceContext& device_ctx,
                                                   uint64_t size,
                                                   VkDeviceMemory handle)
    : VulkanAllocation(AllocationType::kImported, device_ctx, size, handle) {}

VulkanMemoryMgr::VulkanMemoryMgr(const VulkanDeviceContext& device)
    : device_{device}
    , mem_properties_{}
    , total_allocated_{0ul}
    , peak_total_allocated_{0ul}
    , total_imported_{0ul}
    , peak_total_imported_{0ul}
    , ext_mem_type_index_{0}
    , memory_heap_index_{0}
    , mem_properties_2_{}
    , mem_budget_properties_{} {
  vkGetPhysicalDeviceMemoryProperties(device_.getPhysicalDevice().getHandle(),
                                      &mem_properties_);

  // Get the memory type to use for external memory allocations
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = 256;  // TODO(scb): can this be empty?
  buffer_info.usage =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  auto const device_handle = device_.getHandle();
  VkBuffer buffer = VK_NULL_HANDLE;
  auto result = vkCreateBuffer(device_handle, &buffer_info, nullptr, &buffer);
  CHECK_VKRESULT(result, "Creating temporary buffer to get memory type bits");

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_handle, buffer, &req);
  ext_mem_type_index_ = find_memory_type_with_properties(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_properties_);

  memory_heap_index_ = mem_properties_.memoryTypes[ext_mem_type_index_].heapIndex;
  VLOG(1) << "Using memory heap index: " << memory_heap_index_;
  // Initialize physical device memory budget properties structures variables
  mem_budget_properties_.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
  mem_budget_properties_.pNext = nullptr;
  // Initialize physical device memory properties structure variables
  mem_properties_2_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  mem_properties_2_.pNext = &mem_budget_properties_;

  if (buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_handle, buffer, nullptr);
  }
}

VulkanMemoryMgr::~VulkanMemoryMgr() {
  freeAllAllocations();
}

VulkanMemoryMgr::allocation_ptr VulkanMemoryMgr::alloc(
    std::string_view resource_name,
    const VkMemoryRequirements& requirements,
    VkMemoryPropertyFlags properties,
    bool enable_export,
    bool enable_device_address,
    int32_t import_allocation_fd,
    std::optional<LoggingCallback> logging_callback) {
  std::lock_guard<std::mutex> auto_lock(allocation_lock_);

  CHECK(!(enable_export && import_allocation_fd >= 0))
      << "Export and Import are mutually-exclusive";

  VkMemoryAllocateInfo alloc_info = {};
  VkStructChainBuilder alloc_info_chain(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                        &alloc_info);
  alloc_info.allocationSize = requirements.size;
  alloc_info.memoryTypeIndex = find_memory_type_with_properties(
      requirements.memoryTypeBits, properties, mem_properties_);
  VkMemoryAllocateFlagsInfo flags_info = {};
  if (enable_device_address) {
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    alloc_info_chain.add(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &flags_info);
  }

  VkExportMemoryAllocateInfo export_info = {};
  VkImportMemoryFdInfoKHR import_info = {};
  if (enable_export) {
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    alloc_info_chain.add(VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &export_info);
  } else if (import_allocation_fd >= 0) {
    import_info.fd = import_allocation_fd;
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    alloc_info_chain.add(VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &import_info);
  }

  VkDeviceMemory mem_handle;
  auto result = vkAllocateMemory(device_.getHandle(), &alloc_info, nullptr, &mem_handle);
  CHECK_OOM_VKRESULT(result,
                     "Allocating memory",
                     resource_name,
                     requirements.size,
                     device_,
                     logging_callback);

  std::unique_ptr<VulkanAllocation> allocation;
  if (enable_export) {
    allocation = std::make_unique<VulkanExportableAllocation>(
        device_, requirements.size, mem_handle);
  } else if (import_allocation_fd >= 0) {
    allocation = std::make_unique<VulkanImportedAllocation>(
        device_, requirements.size, mem_handle);
  } else {
    allocation =
        std::make_unique<VulkanAllocation>(device_, requirements.size, mem_handle);
  }

  auto [itr, inserted] = allocation_map_.try_emplace(mem_handle, std::move(allocation));

  CHECK(inserted) << device_.getGpuId() << ':' << "0x" << std::hex << mem_handle << ":"
                  << std::dec << requirements.size;

  if (import_allocation_fd >= 0) {
    total_imported_ += itr->second->size();
    if (total_imported_ > peak_total_imported_) {
      peak_total_imported_ = total_imported_;
    }
  } else {
    total_allocated_ += itr->second->size();
    if (total_allocated_ > peak_total_allocated_) {
      peak_total_allocated_ = total_allocated_;
    }
  }

  // The sinked ptr is what we hand out to callers. It does not own the allocation.
  return make_sinked_ptr<VulkanAllocation, VulkanMemoryMgr>(itr->second.get());
}

void VulkanMemoryMgr::free(allocation_ptr allocation) {
  std::lock_guard<std::mutex> auto_lock(allocation_lock_);

  auto& deleter = allocation.get_deleter();
  CHECK(typeid(deleter) == typeid(SinkedDeleter<VulkanAllocation, VulkanMemoryMgr>));
  deleter.allow_delete_ = true;

  auto handle = allocation->getHandle();
  allocation.reset();

  if (handle != VK_NULL_HANDLE) {
    auto it = allocation_map_.find(handle);
    if (it != allocation_map_.end()) {
      vkFreeMemory(device_.getHandle(), handle, nullptr);
      if (it->second->getAllocationType() ==
          VulkanAllocation::AllocationType::kImported) {
        total_imported_ -= it->second->size();
      } else {
        total_allocated_ -= it->second->size();
      }
      allocation_map_.erase(handle);
    } else {
      THROW_RUNTIME_EX("Failed to find allocation block in allocation map");
    }
  } else {
    // TODO(scb): check/throw?
    LOG(WARNING) << "Unable to free untracked memory block, double free?";
  }
}

void VulkanMemoryMgr::freeAllAllocations() {
#if COUT_DEBUG
  std::cout << "freeing all vulkan allocations" << std::endl;
#endif
  std::lock_guard<std::mutex> auto_lock(allocation_lock_);
  if (allocation_map_.empty()) {
#if COUT_DEBUG
    std::cout << "allocation map is empty" << std::endl;
#endif
    return;
  }

  // We should never get here as owners of the allocated blocks should have already freed
  // them. If we get to this point something is likely leaking allocation blocks
  LOG(WARNING)
      << "Vulkan memory manager has outstanding allocations on shutdown for device "
      << device_.getGpuUUID();
  for (auto& allocation : allocation_map_) {
#if COUT_DEBUG
    std::cout << "Freeing " << allocation.second->size() << " bytes" << std::endl;
#endif
    vkFreeMemory(device_.getHandle(), allocation.first, nullptr);
  }
  total_allocated_ = 0ul;
  total_imported_ = 0ul;

  // TODO: verify counts and sizes are zero!
  allocation_map_.clear();
}

uint64_t VulkanMemoryMgr::getPeakMemoryUsage() const {
  return peak_total_allocated_;
}

MemoryBudgetInfo VulkanMemoryMgr::getMemoryBudget() {
  MemoryBudgetInfo budget = {};
  vkGetPhysicalDeviceMemoryProperties2(device_.getPhysicalDeviceHandle(),
                                       &mem_properties_2_);
  budget.used = mem_budget_properties_.heapUsage[memory_heap_index_];
  budget.total = mem_budget_properties_.heapBudget[memory_heap_index_];
  budget.available = budget.total - budget.used;
  return budget;
}

void VulkanMemoryMgr::logMemoryBudgetInfo(std::ostream& os) {
  vkGetPhysicalDeviceMemoryProperties2(device_.getPhysicalDeviceHandle(),
                                       &mem_properties_2_);

  auto const& heap_aggregate_property_flags =
      device_.getPhysicalDevice().getMemoryHeapProperties();
  auto heap_count = heap_aggregate_property_flags.size();

  StreamStatFormatter log_stat(os);
  log_stat << "Memory heap summary for gpu " << device_.getGpuId() << "\n";
  for (uint32_t i = 0; i < heap_count; i++) {
    log_stat << "\n-- Heap " << i << " ["
             << memory_heap_flag_to_string(
                    mem_properties_2_.memoryProperties.memoryHeaps[i].flags,
                    heap_aggregate_property_flags[i])
             << "] --\n";
    log_stat.memory_stat("total", mem_budget_properties_.heapBudget[i]);
    log_stat.memory_stat("used", mem_budget_properties_.heapUsage[i]);
    log_stat.memory_stat(
        "available",
        mem_budget_properties_.heapBudget[i] - mem_budget_properties_.heapUsage[i]);
  }
}

}  // namespace gfx
