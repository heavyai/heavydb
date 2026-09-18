/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "GfxDriver/Resources/Types.h"

namespace gfx {

// Container for referencing a VertexBuffer and offset into the buffer that can
// be passed to standard containers

struct VertexBufferRef {
  // Support all construction except default construction as we don't want an
  // empty VertexBuffer reference. Can be constructed with via initializer_list
  VertexBufferRef(const VertexBuffer& vertex_buffer, uint64_t offset_bytes)
      : vertex_buffer{vertex_buffer}, offset_bytes{offset_bytes} {}
  VertexBufferRef(const VertexBufferRef&) = default;
  VertexBufferRef(VertexBufferRef&&) = default;
  VertexBufferRef& operator=(const VertexBufferRef&) = default;
  VertexBufferRef& operator=(VertexBufferRef&&) = default;

  // Reference to the VertexBuffer
  std::reference_wrapper<const VertexBuffer> vertex_buffer;

  // Offset in bytes into the base Buffer where the vertex data begins
  uint64_t offset_bytes;
};

using VertexBufferRefs = std::vector<VertexBufferRef>;

}  // namespace gfx
