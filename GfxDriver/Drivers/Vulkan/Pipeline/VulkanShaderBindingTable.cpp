/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanShaderBindingTable.h"

#include <cstring>
#include <iostream>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanMemoryUtils.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "Shared/DebugOutputStream.h"

#define DEBUG_PRINT_SBT_INFO false
#define SBT_DEBUG_PRINT() DEBUG_OUTPUT_STREAM(DEBUG_PRINT_SBT_INFO, std::cout)

namespace gfx {

namespace {
// Determine the number of items in each SBT entry region by mapping
// ShaderStages in the ShaderGroup to the correct SBT entry
using ShaderCounts = std::array<uint32_t, ShaderBindingTable::Entry::kCount>;
ShaderCounts get_shader_counts(
    const std::vector<RaytracingPipeline::ShaderGroup>& shader_groups) {
  ShaderCounts counts;
  counts.fill(0u);
  for (auto const& group : shader_groups) {
    counts[group.sbt_entry]++;
  }
  return counts;
}
}  // namespace

VulkanShaderBindingTable::VulkanShaderBindingTable(const RaytracingPipeline& pipeline)
    : ShaderBindingTable(), device_{pipeline.getDeviceContext()} {
  auto const& shader_groups = pipeline.getShaderGroups();
  auto shader_counts = get_shader_counts(shader_groups);

  CHECK_EQ(shader_counts[kRayGen], 1u)
      << "There must be exactly one raygen shader in a RaytracingPipeline";

  // Get shader group handle size and alignment info
  auto const& limits = device_.getLimits();
  auto const base_alignment = limits.shader_group_base_alignment;
  auto const handle_size = limits.shader_group_handle_size;
  auto const handle_size_aligned =
      align_up(handle_size, limits.shader_group_handle_alignment);

  // Compute buffer region sizes and strides
  regions_[kRayGen].stride = base_alignment;
  regions_[kRayGen].size = base_alignment;  // There can only be one!
  regions_[kMiss].stride = handle_size_aligned;
  regions_[kMiss].size =
      align_up(shader_counts[kMiss] * handle_size_aligned, base_alignment);
  regions_[kHit].stride = handle_size_aligned;
  regions_[kHit].size =
      align_up(shader_counts[kHit] * handle_size_aligned, base_alignment);
  regions_[kCallable].stride = handle_size_aligned;
  regions_[kCallable].size =
      align_up(shader_counts[kCallable] * handle_size_aligned, base_alignment);

  // Compute total buffer size and base offsets into the buffer for each SBT entry
  std::array<uint32_t, Entry::kCount> offsets;
  offsets.fill(0u);
  uint64_t buffer_size{0u};
  uint32_t index = 0;
  for (auto const& r : regions_) {
    offsets[index++] = buffer_size;
    buffer_size += r.size;
  }

  auto& resource_mgr = device_.getResourceManager();
  // Create the buffer to store the SBT
  // Use a host visible buffer to simplify writing the table entries
  // The SBT is very small and should be cached on the GPU immediately on use
  buffer_ = resource_mgr.createHostVisibleBuffer("ShaderBindingTable",
                                                 {BufferType::kShaderBindingTableBuffer,
                                                  buffer_size,
                                                  BufferUsageBits::kDeviceAddressBit});

  // Get the base device address of the SBT
  auto buffer_address = buffer_->getSourceBufferWrapper().getDeviceAddress();

  index = 0;
  for (auto& r : regions_) {
    // Use the offsets to compute the buffer address for each SBT region
    if (r.size > 0u) {
      r.address = buffer_address + offsets[index];
    }
    index++;
  }

  // Loop over all the ShaderGroups, copying the shader handles into their slots
  auto* data = reinterpret_cast<uint8_t*>(buffer_->map());
  CHECK(data);

  for (auto const& group : shader_groups) {
    auto entry = group.sbt_entry;
    SBT_DEBUG_PRINT() << "SBT Offset: " << offsets[entry] << "  " << group << std::endl;

    // Copy the handle data
    CHECK_LE(offsets[entry] + handle_size, buffer_size);
    auto* handle_data = group.handle_data.data();
    CHECK(handle_data);
    std::memcpy(data + offsets[entry], handle_data, handle_size);
    // Add handle_size_aligned to the offset in case there are multiple shaders
    // of that entry type in the SBT
    offsets[entry] += handle_size_aligned;
  }

  buffer_->unmap();
}

VulkanShaderBindingTable::~VulkanShaderBindingTable() {
  auto& resource_mgr = device_.getResourceManager();
  if (buffer_) {
    resource_mgr.destroyHostVisibleBuffer(std::move(buffer_));
  }
}

const StridedDeviceAddressRegion& VulkanShaderBindingTable::getRegion(Entry entry) const {
  return regions_[entry];
}

}  // namespace gfx
