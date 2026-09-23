/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/ShaderBindingTable.h"

#include <array>

#include <vulkan/vulkan.h>

#include "GfxDriver/Pipeline/Pipeline.h"

namespace gfx {

class VulkanShaderBindingTable : public ShaderBindingTable {
 public:
  explicit VulkanShaderBindingTable(const RaytracingPipeline& pipeline);
  ~VulkanShaderBindingTable() override;

  const StridedDeviceAddressRegion& getRegion(Entry entry) const override;

 private:
  const DeviceContext& device_;
  std::array<StridedDeviceAddressRegion, Entry::kCount> regions_;
  HostVisibleBufferWrapperUqPtr buffer_;
};

}  // namespace gfx
