/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferWrapper.h"
#include "Logger/Logger.h"

namespace gfx {

struct BufferWrapperInternalsAccessor {
 protected:
  static Buffer& getUnderlyingBuffer(BufferWrapper& buffer_wrapper) {
    CHECK(buffer_wrapper.buffer_);
    return *(buffer_wrapper.buffer_);
  }

  static const Buffer& getUnderlyingBuffer(const BufferWrapper& buffer_wrapper) {
    CHECK(buffer_wrapper.buffer_);
    return *(buffer_wrapper.buffer_);
  }
};

}  // namespace gfx
