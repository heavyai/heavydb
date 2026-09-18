/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/PixelBuffer2d.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace QueryRenderer {

template <typename T>
class QueryIdMapPixelBuffer {
 public:
  QueryIdMapPixelBuffer(gfx::ResourceManager& rsrcMgr, uint32_t width, uint32_t height)
      : resource_manager_(rsrcMgr) {
    _init(rsrcMgr, width, height);
  }

  ~QueryIdMapPixelBuffer() {
    if (pbo_) {
      resource_manager_.destroyBuffer(std::move(pbo_));
    }
  }

  uint32_t getWidth() const { return getPixelBuffer2d().getWidth(); }

  uint32_t getHeight() const { return getPixelBuffer2d().getHeight(); }

  void resize(uint32_t width, uint32_t height) {
    getPixelBuffer2d().resize(width, height);
  }

  void readIdBuffer(uint32_t width, uint32_t height, T* idBuffer);

  inline gfx::PixelBuffer2d& getPixelBuffer2d() const {
    CHECK(pbo_);
    return *static_cast<gfx::PixelBuffer2d*>(pbo_.get());
  }

 private:
  void _init(::gfx::ResourceManager& rsrcMgr, uint32_t width, uint32_t height);

  gfx::BufferWrapperUqPtr pbo_;
  gfx::ResourceManager& resource_manager_;
};

template <>
void QueryIdMapPixelBuffer<uint32_t>::_init(gfx::ResourceManager& rsrcMgr,
                                            uint32_t width,
                                            uint32_t height);

template <>
void QueryIdMapPixelBuffer<uint32_t>::readIdBuffer(uint32_t width,
                                                   uint32_t height,
                                                   uint32_t* idBuffer);

template <>
void QueryIdMapPixelBuffer<int32_t>::_init(gfx::ResourceManager& rsrcMgr,
                                           uint32_t width,
                                           uint32_t height);

template <>
void QueryIdMapPixelBuffer<int32_t>::readIdBuffer(uint32_t width,
                                                  uint32_t height,
                                                  int32_t* idBuffer);

}  // namespace QueryRenderer
