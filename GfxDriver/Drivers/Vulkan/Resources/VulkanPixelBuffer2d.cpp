/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanPixelBuffer2d.h"

#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/RenderError.h"

namespace gfx {

VulkanPixelBuffer2d::VulkanPixelBuffer2d(std::string_view resource_tracking_string,
                                         BufferAllocatorShPtr buffer_allocator,
                                         uint32_t width,
                                         uint32_t height,
                                         const PixelFormat pixel_format,
                                         void* data)
    : PixelBuffer2d(resource_tracking_string,
                    std::move(buffer_allocator),
                    width,
                    height,
                    pixel_format,
                    data)
    , cmd_buffer_{nullptr} {}

VulkanPixelBuffer2d::~VulkanPixelBuffer2d() {
  // Dangling command buffers are currently cleaned up by StagingContext destructor
  // TODO(scb) This should be made more robust as part of async ID readback scheduling
  // fixes in Renderer
}

void VulkanPixelBuffer2d::finishCommandBuffer() {
  if (cmd_buffer_) {
    auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
    auto& staging = vk_device.getStagingContext();
    staging.copyTextureToPixelBufferFinish(*this);
    cmd_buffer_ = nullptr;
  }
}

void VulkanPixelBuffer2d::setCommandBuffer(VulkanCommandBuffer* cmd_buffer) {
  // validate
  CHECK(cmd_buffer != nullptr);
  // finish and return any existing CommandBuffer
  finishCommandBuffer();
  // capture new CommandBuffer
  cmd_buffer_ = cmd_buffer;
}

VulkanCommandBuffer* VulkanPixelBuffer2d::releaseCommandBuffer() {
  // getting should unset
  // returns null on repeat calls
  auto rtn = cmd_buffer_;
  cmd_buffer_ = nullptr;
  return rtn;
}

void VulkanPixelBuffer2d::readPixels(uint32_t width,
                                     uint32_t height,
                                     const PixelFormat pixel_format,
                                     void* data) {
  BufferWrapper::validateUsability(__FILE__, __LINE__);

  CHECK(pixel_format_ == pixel_format);

  RUNTIME_EX_ASSERT(
      width == width_ && height == height_,
      "Invalid dimensions of data buffer to read pixels into. The data buffer is " +
          std::to_string(width) + "x" + std::to_string(height) +
          ", but the pixel buffer is " + std::to_string(width_) +
          std::to_string(height_));

  // finish the asynchronous image-to-buffer transfer
  // this just finishes the command buffer and returns it to the pool
  // no-op on repeat calls
  auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto& staging = vk_device.getStagingContext();
  staging.copyTextureToPixelBufferFinish(*this);

  // extract the buffer contents
  // returns the same data on repeat calls
  const uint64_t num_image_bytes = width * height * pixelFormatDataSize(pixel_format);
  BufferWrapper::getData(data, num_image_bytes);
}

}  // namespace gfx
