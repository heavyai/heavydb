/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferWrapper.h"
#include "GfxDriver/Resources/Enums.h"

namespace gfx {

class PixelBuffer2d : public BufferWrapper {
 public:
  explicit PixelBuffer2d(std::string_view resource_tracking_string,
                         BufferAllocatorShPtr buffer_allocator,
                         uint32_t width,
                         uint32_t height,
                         const PixelFormat pixel_format,
                         void* data = nullptr);
  ~PixelBuffer2d() override = default;

  uint32_t getWidth() const { return width_; }
  uint32_t getHeight() const { return height_; }
  PixelFormat getPixelFormat() const { return pixel_format_; }

  void resize(uint32_t width, uint32_t height, void* data = nullptr);

  virtual void readPixels(uint32_t width,
                          uint32_t height,
                          const PixelFormat pixel_format,
                          void* data) = 0;

 protected:
  uint32_t width_;
  uint32_t height_;
  PixelFormat pixel_format_;
};

}  // namespace gfx
