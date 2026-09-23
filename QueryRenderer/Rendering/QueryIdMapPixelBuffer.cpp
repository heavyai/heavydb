/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/QueryIdMapPixelBuffer.h"

#include <GfxDriver/Enums.h>

using ::gfx::PixelFormat;

namespace QueryRenderer {

template <>
void QueryIdMapPixelBuffer<uint32_t>::_init(gfx::ResourceManager& rsrcMgr,
                                            uint32_t width,
                                            uint32_t height) {
  pbo_ = rsrcMgr.createPixelBuffer("ID PBO", width, height, PixelFormat::kR32UI);
}

template <>
void QueryIdMapPixelBuffer<uint32_t>::readIdBuffer(uint32_t width,
                                                   uint32_t height,
                                                   uint32_t* idBuffer) {
  getPixelBuffer2d().readPixels(width, height, PixelFormat::kR32UI, idBuffer);
}

}  // namespace QueryRenderer
