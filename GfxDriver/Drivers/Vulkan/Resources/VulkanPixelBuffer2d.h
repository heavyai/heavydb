/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/PixelBuffer2d.h"

namespace gfx {

class VulkanCommandBuffer;

/*
  PixelBuffers are only used to read back pixels from the gpu (pixel pack buffer).
  Ultimately this class will be subsumed by a more general purpose StagingBuffer
  class. For now we'll keep things as simple as possible for our current usage.
*/
class VulkanPixelBuffer2d : public PixelBuffer2d {
 public:
  explicit VulkanPixelBuffer2d(std::string_view resource_tracking_string,
                               BufferAllocatorShPtr buffer_allocator,
                               uint32_t width,
                               uint32_t height,
                               const PixelFormat pixel_format,
                               void* data = nullptr);
  ~VulkanPixelBuffer2d() override;

  void setCommandBuffer(VulkanCommandBuffer* cmd_buffer);
  VulkanCommandBuffer* releaseCommandBuffer();

  void readPixels(uint32_t width,
                  uint32_t height,
                  const PixelFormat pixel_format,
                  void* data) override;

 private:
  VulkanCommandBuffer* cmd_buffer_;

  void finishCommandBuffer();
};

}  // namespace gfx
