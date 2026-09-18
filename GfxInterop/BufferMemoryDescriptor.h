/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace gfx {

struct BufferMemoryDescriptor {
  // TODO(croot): consider using std::byte* here when we move to CUDA 11, which has c++17
  // support. See: https://omnisci.atlassian.net/browse/BE-5248

  // TODO(croot): consider adding a pointer to a LayoutManager if the mapped buffer has a
  // layout
  int8_t* handle = nullptr;
  uint64_t num_bytes = 0;
};

}  // namespace gfx
