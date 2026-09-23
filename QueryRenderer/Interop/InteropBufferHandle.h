/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxInterop/BufferMemoryDescriptor.h"

namespace QueryRenderer {

using gfx::BufferMemoryDescriptor;

struct PolyBufferMemoryDescriptors {
  BufferMemoryDescriptor vbo_descriptor;
  BufferMemoryDescriptor line_indirect_vbo_descriptor;
  BufferMemoryDescriptor poly_indirect_vbo_descriptor;
  BufferMemoryDescriptor ssbo_descriptor;
  BufferMemoryDescriptor poly_rowids_ssbo_descriptor;
};

struct LineBufferMemoryDescriptors {
  BufferMemoryDescriptor vbo_descriptor;
  BufferMemoryDescriptor line_indirect_vbo_descriptor;
  BufferMemoryDescriptor ssbo_descriptor;
};

struct RasterMeshBufferMemoryDescriptors {
  BufferMemoryDescriptor vbo_descriptor;
  BufferMemoryDescriptor ibo_descriptor;
};

}  // namespace QueryRenderer
