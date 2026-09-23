/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <glm/vec2.hpp>

#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"

namespace QueryRenderer {

enum SymbolFlagBits {
  kNone = 0,
  kXYSymmetry = 1 << 0,
  kXSymmetry = 1 << 1,
  kCircle = 1 << 2,
  kTriangle = 1 << 3,
  kUseWindingNumberTest = 1 << 4
};

struct SymbolDefinitions {
  bool initialized = false;
  std::vector<uint32_t> flags;
  std::vector<uint32_t> segment_counts;
  std::vector<uint32_t> vert_offsets;
  std::vector<glm::vec2> verts;

  void resize(size_t size) {
    flags.resize(size);
    segment_counts.resize(size);
    vert_offsets.resize(size);
    verts.clear();
    initialized = false;
  }
};

void init_symbol_defs(SymbolDefinitions& symbol_defs);

void generate_fast_symbol_interface_blocks(
    gfx::GlslStructBuilder& fragment_inputs,
    std::optional<gfx::GlslStructBuilder>& geometry_inputs,
    bool is_angle_uniform,
    bool has_accumulator,
    bool use_mesh_shader);

}  // namespace QueryRenderer
