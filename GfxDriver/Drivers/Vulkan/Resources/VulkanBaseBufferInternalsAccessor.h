/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanBaseBuffer.h"

namespace gfx {

struct VulkanBaseBufferInternalsAccessor {
 protected:
  static VulkanMemoryMgr::allocation_ptr& getVulkanAllocation(VulkanBaseBuffer& buffer) {
    return buffer.vulkan_allocation_;
  }

  static const VulkanMemoryMgr::allocation_ptr& getVulkanAllocation(
      const VulkanBaseBuffer& buffer) {
    return buffer.vulkan_allocation_;
  }
};

}  // namespace gfx
