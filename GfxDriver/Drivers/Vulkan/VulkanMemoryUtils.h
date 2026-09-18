/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Utils/LoggingUtils.h"

namespace gfx {

// Iterate over the various memory types in mem_properties, finding a type
// that satisfies required_props
uint32_t find_memory_type_with_properties(
    uint32_t mem_type_bits_requirement,
    VkMemoryPropertyFlags required_props,
    VkPhysicalDeviceMemoryProperties& mem_properties);

// Get memory property flags (memory type bits) supported by each heap
// For example will return HOST_VISIBLE and DEVICE_LOCAL bits for the
// BAR region heap
std::vector<VkMemoryPropertyFlags> get_aggregate_memory_heap_props(
    const VkPhysicalDeviceMemoryProperties& mem_props);

// Create a string for a heap flag bit
std::string memory_heap_flag_to_string(VkMemoryHeapFlags flags,
                                       VkMemoryPropertyFlags props);

// Write the current memory budget info to a stream
// Output total, used, available (free) for each heap
void log_memory_budget(const DeviceContext& device, std::ostream& os);

// Align a value to the specified alignment by rounding up if necessary
// T must be an unsigned integral type (uint32_t, uint64_t, size_t)
template <typename T>
inline const T align_up(T value, T alignment) noexcept {
  if constexpr (std::is_unsigned_v<T> && std::is_arithmetic_v<T>) {
    return (value + alignment - 1) & ~(alignment - 1);
  } else {
    static_assert(!(std::is_unsigned_v<T> && std::is_arithmetic_v<T>),
                  "align_up must only be used with unsigned integral types");
  }
}

// class VulkanGpuMemoryHog
//
// Helper class used to consume all gpu memory to ensure the next allocation
// attempt will fail. Used to trigger OOM errors during testing
//
// Constructor attempts to allocate all gpu memory except reserved_bytes
// Memory is allocated in progressively smaller blocks down to 64 bytes
// Memory blocks are freed on destruction
// Allocation info will be written to log_stream
// Note: It will typically over shoot the mark a bit due to rounding up allocation
// sizes to accomodate alignment
class VulkanGpuMemoryHog {
 public:
  VulkanGpuMemoryHog(const VulkanDeviceContext& device,
                     uint64_t reserved_bytes,
                     std::ostream& log_stream = null_ostream);
  ~VulkanGpuMemoryHog();

 private:
  const VulkanDeviceContext& device_;
  std::vector<VkDeviceMemory> memory_blocks_;
};

}  // namespace gfx
