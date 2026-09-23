/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"

#include <glm/vec2.hpp>

namespace QueryRenderer {

static constexpr uint32_t kMaxWindBarbSpeed = 150.0;

struct WindBarbDefinitions {
  bool initialized = false;
  std::vector<uint32_t> barb_counts;
  std::vector<uint32_t> barb_offsets;
  std::vector<uint32_t> pennant_counts;
  std::vector<uint32_t> pennant_offsets;
  std::vector<glm::vec2> verts;

  void resize(size_t size) {
    barb_counts.resize(size);
    barb_offsets.resize(size);
    pennant_counts.resize(size);
    pennant_offsets.resize(size);
    verts.clear();
    initialized = false;
  }
};

void init_wind_barb_defs(WindBarbDefinitions& wind_barb_defs);
void generate_wind_barb_interface_blocks(gfx::GlslStructBuilder& geometry_inputs,
                                         gfx::GlslStructBuilder& fragment_inputs,
                                         bool is_direction_uniform);

}  // namespace QueryRenderer
