/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Enums.h"

namespace gfx {

struct BufferCreateInfo {
  BufferType buffer_type = BufferType::kUnspecified;
  uint64_t size = 0;
  BufferUsageBits usage = BufferUsageBits::kNone;
  BufferAccessType access_type = BufferAccessType::kDeviceLocal;
  int32_t import_allocation_fd = -1;
  IndexBufferDataType index_buffer_data_type = IndexBufferDataType::kUnsigned32;
};

}  // namespace gfx
