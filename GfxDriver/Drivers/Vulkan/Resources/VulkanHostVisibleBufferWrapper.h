/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanBaseBuffer.h"
#include "GfxDriver/Resources/HostVisibleBufferWrapper.h"

namespace gfx {

class VulkanHostVisibleBufferWrapper : public HostVisibleBufferWrapper {
 public:
  VulkanHostVisibleBufferWrapper(BufferWrapperUqPtr source_buffer_wrapper);
  VulkanHostVisibleBufferWrapper() = delete;
  ~VulkanHostVisibleBufferWrapper() override;

  void map(void** data) override;
  void* map() override;
  void unmap() override;

  static void mapImpl(VulkanBaseBuffer& vk_buffer, void** data);
  static void unmapImpl(VulkanBaseBuffer& vk_buffer);
};

}  // namespace gfx
