/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryUtils.h"

#include <iostream>

#include "GfxDriver/Drivers/Vulkan/VulkanPhysicalDevice.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

uint32_t find_memory_type_with_properties(
    uint32_t mem_type_bits_requirement,
    VkMemoryPropertyFlags required_props,
    VkPhysicalDeviceMemoryProperties& mem_properties) {
  for (uint32_t memory_index = 0; memory_index < mem_properties.memoryTypeCount;
       ++memory_index) {
    const uint32_t memory_type_bits = (1 << memory_index);
    const bool is_required_type = mem_type_bits_requirement & memory_type_bits;
    const VkMemoryPropertyFlags properties =
        mem_properties.memoryTypes[memory_index].propertyFlags;
    const bool has_required_properties = (properties & required_props) == required_props;

    if (is_required_type && has_required_properties) {
      return memory_index;
    }
  }
  LOG(FATAL) << "Failed to find suitable device memory type";
  return 0;
}

std::vector<VkMemoryPropertyFlags> get_aggregate_memory_heap_props(
    const VkPhysicalDeviceMemoryProperties& mem_props) {
  auto heap_count = mem_props.memoryHeapCount;
  auto type_count = mem_props.memoryTypeCount;
  std::vector<VkMemoryPropertyFlags> heap_aggregate_property_flags(heap_count, 0);
  for (uint32_t i = 0; i < type_count; ++i) {
    auto const& type = mem_props.memoryTypes[i];
    heap_aggregate_property_flags[type.heapIndex] |= type.propertyFlags;
  }
  return heap_aggregate_property_flags;
}

std::string memory_heap_flag_to_string(VkMemoryHeapFlags flags,
                                       VkMemoryPropertyFlags props) {
  switch (flags) {
    case VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
      if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        return "Device,Host";
      } else {
        return "Device";
      }
    case VK_MEMORY_HEAP_MULTI_INSTANCE_BIT:
      return "Multi-Instance";
    default:
      return "Host";
  }
}

void log_memory_budget(const DeviceContext& device, std::ostream& os) {
  auto budget = device.getMemoryBudget();
  os << "Memory state -- total: " << budget.total << "  used: " << budget.used
     << "  available: " << budget.available << "\n";
}

//
// VulkanGpuMemoryHog
//
VulkanGpuMemoryHog::VulkanGpuMemoryHog(const VulkanDeviceContext& device,
                                       uint64_t reserved_bytes,
                                       std::ostream& log_stream)
    : device_{device} {
  const uint64_t kMinAllocationSize = 64;
  auto budget = device.getMemoryBudget();

  log_stream << "VulkanGpuMemoryHog  available: " << budget.available
             << "  reserved: " << reserved_bytes << std::endl;

  if (budget.available <= (reserved_bytes + kMinAllocationSize)) {
    log_stream << "Device already full" << std::endl;
    return;
  }

  // initial allocation size
  auto alloc_size = budget.available - reserved_bytes;

  VkPhysicalDeviceMemoryProperties mem_properties{};
  vkGetPhysicalDeviceMemoryProperties(device_.getPhysicalDevice().getHandle(),
                                      &mem_properties);

  // Create a buffer so we can get the memory type bits
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = alloc_size;
  buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  auto const device_handle = device_.getHandle();
  VkBuffer buffer = VK_NULL_HANDLE;
  auto result = vkCreateBuffer(device_handle, &buffer_info, nullptr, &buffer);
  CHECK_VKRESULT(result, "Creating temporary buffer to get memory type bits");

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_handle, buffer, &req);
  auto alignment = req.alignment;
  alloc_size = req.size;
  vkDestroyBuffer(device_handle, buffer, nullptr);
  VkMemoryPropertyFlags required_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.memoryTypeIndex = find_memory_type_with_properties(
      req.memoryTypeBits, required_props, mem_properties);

  // Hoover it up
  log_stream << "  Hoovering..." << std::endl;
  while (budget.available > reserved_bytes && alloc_size >= kMinAllocationSize) {
    alloc_info.allocationSize = alloc_size;
    VkDeviceMemory mem_handle = VK_NULL_HANDLE;
    auto result = vkAllocateMemory(device_handle, &alloc_info, nullptr, &mem_handle);
    if (result == VK_SUCCESS) {
      memory_blocks_.push_back(mem_handle);
      budget = device.getMemoryBudget();
      log_stream << "    allocated: " << alloc_size << std::endl;
      alloc_size = align_up(budget.available - reserved_bytes, alignment);
    } else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
      alloc_size = align_up(alloc_size / 2, alignment);
      log_stream << "    allocation failed - new size: " << alloc_size << std::endl;
    } else {
      CHECK_VKRESULT(result, "Hoovering memory");
    }
  }

  log_stream << "  Allocated " << memory_blocks_.size() << " memory blocks" << std::endl;
  log_stream << "  Available memory: " << budget.available << std::endl;
  log_memory_budget(device, log_stream);
  log_stream << std::endl;
}

VulkanGpuMemoryHog::~VulkanGpuMemoryHog() {
  for (auto& allocation : memory_blocks_) {
    vkFreeMemory(device_.getHandle(), allocation, nullptr);
  }
}

}  // namespace gfx
