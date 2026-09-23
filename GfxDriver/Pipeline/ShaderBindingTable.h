/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <ostream>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

// StridedDeviceAddressRegion struct required by the traceRays command executor
// This struct must match the layout of the Vulkan struct as it will be reinterpret_cast
// to a VkStridedDeviceAddressRegion by the VulkanCommandExecutor
struct StridedDeviceAddressRegion {
  uint64_t address{0u};
  uint64_t stride{0u};
  uint64_t size{0u};
};

// Helper class to create ShaderBindingTables using the ShaderGroups stored in
// a RaytracingPipeline. This will automatically create and fill the SBT buffer
// with ShaderGroup handle data, and initialize the StridedDeviceAddressRegion structs
// for traceRays command execution
class ShaderBindingTable {
 public:
  enum Entry { kRayGen, kMiss, kHit, kCallable, kCount };
  virtual ~ShaderBindingTable() = default;
  virtual const StridedDeviceAddressRegion& getRegion(Entry entry) const = 0;
};

ShaderBindingTable::Entry shader_stage_to_sbt_entry(const ShaderStage stage);

std::ostream& operator<<(std::ostream& os, const ShaderBindingTable::Entry value);

}  // namespace gfx
