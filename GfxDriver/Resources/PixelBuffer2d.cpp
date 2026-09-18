/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/PixelBuffer2d.h"

#include "GfxDriver/RenderError.h"

namespace gfx {

PixelBuffer2d::PixelBuffer2d(std::string_view resource_tracking_string,
                             BufferAllocatorShPtr buffer_allocator,
                             uint32_t width,
                             uint32_t height,
                             const PixelFormat pixel_format,
                             void* data)
    : BufferWrapper(resource_tracking_string,
                    std::move(buffer_allocator),
                    {BufferType::kPixelBuffer,
                     width * height * pixelFormatDataSize(pixel_format),
                     BufferUsageBits::kNone,
                     BufferAccessType::kHostVisibleCached},
                    std::nullopt)
    , width_(width)
    , height_(height)
    , pixel_format_(pixel_format) {
  if (data) {
    updateSubData(data, getNumBytes(), 0);
  }
}

void PixelBuffer2d::resize(uint32_t width, uint32_t height, void* data) {
  if (width != width_ || height != height_) {
    RUNTIME_EX_ASSERT(width > 0 && height > 0,
                      "Invalid dimensions " + std::to_string(width) + "x" +
                          std::to_string(height) +
                          " for the texture. Dimensions must be > 0");

    rebuild(data, width * height * pixelFormatDataSize(pixel_format_));

    width_ = width;
    height_ = height;
  }
}

}  // namespace gfx
