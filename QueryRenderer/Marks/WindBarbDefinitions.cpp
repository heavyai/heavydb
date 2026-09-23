/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/WindBarbDefinitions.h"

#include <glm/glm.hpp>

namespace QueryRenderer {

void init_wind_barb_defs(WindBarbDefinitions& barb_defs) {
  if (!barb_defs.initialized) {
    auto num_barb_types = (static_cast<uint32_t>(kMaxWindBarbSpeed) + 4) / 5 + 1;
    barb_defs.resize(num_barb_types);

    // barb 0 is calm, and just rendered as 2 circles
    barb_defs.barb_counts[0] = 0;
    barb_defs.barb_offsets[0] = 0;
    barb_defs.pennant_counts[0] = 0;
    barb_defs.pennant_offsets[0] = 0;

    constexpr float kBarbLen = 0.5f;
    constexpr float kStaffX = -0.9f;
    constexpr float kBarbSpacing = 0.13f;
    constexpr float kPennantSpacing = 0.28f;
    constexpr float kBarbAngle = glm::radians(55.f);
    const auto kBarbDir = glm::vec2(sin(kBarbAngle), cos(kBarbAngle));
    const float kPennantHeight = (kBarbDir * kBarbLen).y;

    auto add_vert = [&](const float x, const float y) {
      barb_defs.verts.emplace_back(x, y);
    };

    auto add_barb = [&](const float t, const float len) {
      glm::vec2 v0(kStaffX, t);
      glm::vec2 v1 = v0 + kBarbDir * kBarbLen * len;
      barb_defs.verts.push_back(v0);
      barb_defs.verts.push_back(v1);
    };

    auto add_pennant = [&](const float t) {
      glm::vec2 v0(kStaffX, t - kPennantHeight);
      glm::vec2 v1 = v0 + kBarbDir * kBarbLen;
      glm::vec2 v2(v0.x, v1.y);
      barb_defs.verts.push_back(v0);
      barb_defs.verts.push_back(v1);
      barb_defs.verts.push_back(v2);
    };

    uint32_t wind_speed = 5;
    // distance from anchor to first barb
    constexpr float kStemOffset = kBarbSpacing;

    for (uint32_t type = 1; type < num_barb_types; ++type, wind_speed += 5) {
      uint32_t num_pennants = wind_speed / 50u;
      uint32_t barb_speed = wind_speed - (num_pennants * 50);
      uint32_t num_barbs = barb_speed / 10;

      // Add pennant triangles first
      barb_defs.pennant_counts[type] = num_pennants;
      barb_defs.pennant_offsets[type] = barb_defs.verts.size();

      float start_t;
      if (num_pennants) {
        start_t = (float)num_pennants * kPennantSpacing + kStemOffset;
      } else {
        start_t = (kPennantSpacing - kPennantHeight) - kBarbSpacing + kStemOffset;
      }
      float t = start_t;
      for (uint32_t i = 0; i < num_pennants; i++, t -= kPennantSpacing) {
        add_pennant(t);
      }

      if (num_pennants) {
        t = t + (kPennantSpacing - kPennantHeight) - kBarbSpacing;
      }

      barb_defs.barb_counts[type] = 1 + num_barbs;  // treat stem as another barb
      barb_defs.barb_offsets[type] = barb_defs.verts.size();

      // add barb stem vertices
      add_vert(kStaffX, -0.9f);
      add_vert(kStaffX, start_t);

      for (uint32_t i = 0; i < num_barbs; i++, t -= kBarbSpacing) {
        add_barb(t, 1.0f);
      }
      // add half barb if needed
      // barb as 5 knots is offset specially
      if (barb_speed % 10) {
        add_barb(t - (wind_speed == 5 ? kBarbSpacing : 0.0f), 0.5f);
        barb_defs.barb_counts[type]++;
      }
    }
    barb_defs.initialized = true;
  }
}

void generate_wind_barb_interface_blocks(gfx::GlslStructBuilder& geometry_inputs,
                                         gfx::GlslStructBuilder& fragment_inputs,
                                         bool is_direction_uniform) {
  // Build fragment shader input interface block
  // Also used for vertex output in point mode
  std::vector<gfx::GlslStructBuilder::Qualifier> quals = {
      gfx::GlslStructBuilder::Qualifier::kFlat};

  // flat uint64_t gRowId
  // flat float gPointSize
  // flat vec4 gFillColor
  // flat vec4 gStrokeColor
  // flat float gStrokeWidth
  // flat uint gBarbId (mapped from speed)
  // flat float gDirection (only when non-uniform)
  // flat float gAnchorScale
  geometry_inputs.addMember("gRowId", gfx::BufferAttrType::kUint64, quals);
  geometry_inputs.addMember("gPointSize", gfx::BufferAttrType::kFloat, quals);
  geometry_inputs.addMember("gFillColor", gfx::BufferAttrType::kVec4f, quals);
  geometry_inputs.addMember("gStrokeColor", gfx::BufferAttrType::kVec4f, quals);
  geometry_inputs.addMember("gStrokeWidth", gfx::BufferAttrType::kFloat, quals);
  geometry_inputs.addMember("gBarbId", gfx::BufferAttrType::kUint, quals);
  geometry_inputs.addMember("gAnchorScale", gfx::BufferAttrType::kFloat, quals);
  if (!is_direction_uniform) {
    geometry_inputs.addMember("gDirection", gfx::BufferAttrType::kFloat, quals);
  }

  // flat uint64_t fRowId
  // vec2 fUV
  // flat uint fBarbId
  // flat float fPointScale
  // flat float fPointSize
  // flat float fApproxMaxCoverage
  // flat float fAnchorScale
  // flat vec4 fFillColor
  // flat vec4 fStrokeColor
  // flat float fStrokeWidth
  fragment_inputs.addMember("fRowId", gfx::BufferAttrType::kUint64, quals);
  fragment_inputs.addMember("fUV", gfx::BufferAttrType::kVec2f);
  fragment_inputs.addMember("fBarbId", gfx::BufferAttrType::kUint, quals);
  fragment_inputs.addMember("fPointScale", gfx::BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fPointSize", gfx::BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fApproxMaxCoverage", gfx::BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fAnchorScale", gfx::BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fFillColor", gfx::BufferAttrType::kVec4f, quals);
  fragment_inputs.addMember("fStrokeColor", gfx::BufferAttrType::kVec4f, quals);
  fragment_inputs.addMember("fStrokeWidth", gfx::BufferAttrType::kFloat, quals);
}

}  // namespace QueryRenderer
