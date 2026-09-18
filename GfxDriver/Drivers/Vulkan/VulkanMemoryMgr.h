/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <unordered_map>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Resources/SinkedPtr.h"

namespace gfx {

class VulkanDeviceContext;

/**
 * Basic memory manager.
 *
 * This is a simple, naive device memory manager, that just allocates device memory
 * as it is requested. It makes no effort at creating pools, allocating from pages
 * or anything. It will need to be replaced with a proper page based manager
 * before moving into production. AMD has an open source allocation manager that
 * we should evaluate:
 * https://gpuopen.com/gaming-product/vulkan-memory-allocator/
 * https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/tree/v2.1.0
 *
 * */
struct VulkanAllocation {
 public:
  enum class AllocationType { kGeneric, kExportable, kImported };

  explicit VulkanAllocation(const VulkanDeviceContext& device_ctx,
                            uint64_t size,
                            VkDeviceMemory handle);
  VulkanAllocation() = delete;
  virtual ~VulkanAllocation() = default;

  uint64_t size();

  VkDeviceMemory getHandle() const { return handle_; }

  AllocationType getAllocationType() { return alloc_type_; }

 protected:
  explicit VulkanAllocation(AllocationType alloc_type,
                            const VulkanDeviceContext& device_ctx,
                            uint64_t size,
                            VkDeviceMemory handle);

  AllocationType const alloc_type_;

  const VulkanDeviceContext& device_ctx_;
  uint64_t size_;
  VkDeviceMemory handle_;
};

struct VulkanExportableAllocation : public VulkanAllocation {
 public:
  explicit VulkanExportableAllocation(const VulkanDeviceContext& device_ctx,
                                      uint64_t size,
                                      VkDeviceMemory handle);
  ~VulkanExportableAllocation() override = default;

  int exportHandle(const std::string& buffer_name = "",
                   std::optional<bool> is_dest = std::nullopt);

 private:
  std::atomic<uint32_t> export_count_ = 0u;
  static std::atomic<uint32_t> total_export_count_;
};

struct VulkanImportedAllocation : public VulkanAllocation {
 public:
  explicit VulkanImportedAllocation(const VulkanDeviceContext& device_ctx,
                                    uint64_t size,
                                    VkDeviceMemory handle);
  ~VulkanImportedAllocation() override = default;
};

class VulkanMemoryMgr {
 public:
  using allocation_ptr = sinked_ptr<VulkanAllocation, VulkanMemoryMgr>;

  explicit VulkanMemoryMgr(const VulkanDeviceContext& device);
  ~VulkanMemoryMgr();

  [[nodiscard]] allocation_ptr alloc(
      std::string_view resource_name,
      const VkMemoryRequirements& requirements,
      VkMemoryPropertyFlags properties,
      bool enable_export,
      bool enable_device_address = false,
      int32_t import_allocation_fd = -1,
      std::optional<LoggingCallback> oom_logging_cb = std::nullopt);
  void free(allocation_ptr allocation);

  uint64_t getPeakMemoryUsage() const;
  // Get current memory budget for allocation heap in use
  MemoryBudgetInfo getMemoryBudget();
  // Log budget information for all heaps (called during OOM logging)
  void logMemoryBudgetInfo(std::ostream& os);

 private:
  const VulkanDeviceContext& device_;
  VkPhysicalDeviceMemoryProperties mem_properties_;
  std::mutex allocation_lock_;
  uint64_t total_allocated_;
  uint64_t peak_total_allocated_;
  uint64_t total_imported_;
  uint64_t peak_total_imported_;

  uint32_t ext_mem_type_index_;

  uint32_t memory_heap_index_;
  VkPhysicalDeviceMemoryProperties2 mem_properties_2_;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT mem_budget_properties_;

  using AllocationMap =
      std::unordered_map<VkDeviceMemory, std::unique_ptr<VulkanAllocation>>;
  AllocationMap allocation_map_;

  void freeAllAllocations();
};

}  // namespace gfx
