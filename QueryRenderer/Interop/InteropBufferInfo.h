/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "GfxDriver/Resources/BufferWrapper.h"
#include "GfxInterop/BufferMemoryDescriptor.h"

namespace QueryRenderer {

struct InteropBufferInfo {
  gfx::BufferMemoryDescriptor mapped_buffer_descriptor;
  const gfx::BufferWrapper* layout_buffer;
  const int64_t invalid_key;
};

}  // namespace QueryRenderer
