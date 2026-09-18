/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/SymbolDefinitions.h"

#include "Logger/Logger.h"
#include "QueryRenderer/Marks/Enums.h"

namespace QueryRenderer {

struct SymbolDef {
  uint8_t flags{0u};
  uint8_t segment_count{0u};
  std::vector<glm::vec2> vertices;
};

SymbolDef get_symbol_def(const SymbolShapeType symbol_type) {
  SymbolDef rtn = {};
  switch (symbol_type) {
    case SymbolShapeType::kCircle:
      rtn.flags = SymbolFlagBits::kCircle;
      break;
    case SymbolShapeType::kSquare:
      rtn.flags = SymbolFlagBits::kXYSymmetry;
      rtn.vertices = {
          glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 1.0f), glm::vec2(1.0f, 0.0f)};
      break;
    case SymbolShapeType::kCross:
      rtn.flags = SymbolFlagBits::kXYSymmetry | SymbolFlagBits::kUseWindingNumberTest;
      rtn.vertices = {glm::vec2(0.0f, 1.0f),
                      glm::vec2(0.3333f, 1.0f),
                      glm::vec2(0.3333f, 0.3333f),
                      glm::vec2(1.0f, 0.3333f),
                      glm::vec2(1.0f, 0.0f)};
      break;
    case SymbolShapeType::kDiamond:
      rtn.flags = SymbolFlagBits::kXYSymmetry;
      rtn.vertices = {glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 0.0f)};
      break;
    case SymbolShapeType::kHexagonHoriz:
      rtn.flags = SymbolFlagBits::kXYSymmetry;
      rtn.vertices = {
          glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 0.5f), glm::vec2(1.0f, 0.0f)};
      break;
    case SymbolShapeType::kHexagonVert:
      rtn.flags = SymbolFlagBits::kXYSymmetry;
      rtn.vertices = {
          glm::vec2(0.0f, 1.0f), glm::vec2(0.5f, 1.0f), glm::vec2(1.0f, 0.0f)};
      break;
    case SymbolShapeType::kTriangleUp:
      rtn.flags = SymbolFlagBits::kTriangle;
      rtn.vertices = {glm::vec2(0.0f, 1.0f),
                      glm::vec2(-1.0f, -1.0f),
                      glm::vec2(1.0f, -1.0f),
                      glm::vec2(0.0f, 1.0f)};
      break;
    case SymbolShapeType::kTriangleDown:
      rtn.flags = SymbolFlagBits::kTriangle;
      rtn.vertices = {glm::vec2(0.0f, -1.0f),
                      glm::vec2(-1.0f, 1.0f),
                      glm::vec2(1.0f, 1.0f),
                      glm::vec2(0.0f, -1.0f)};
      break;
    case SymbolShapeType::kTriangleLeft:
      rtn.flags = SymbolFlagBits::kTriangle;
      rtn.vertices = {glm::vec2(-1.0f, 0.0f),
                      glm::vec2(1.0f, -1.0f),
                      glm::vec2(1.0f, 1.0f),
                      glm::vec2(-1.0f, 0.0f)};
      break;
    case SymbolShapeType::kTriangleRight:
      rtn.flags = SymbolFlagBits::kTriangle;
      rtn.vertices = {glm::vec2(1.0f, 0.0f),
                      glm::vec2(-1.0f, 1.0f),
                      glm::vec2(-1.0f, -1.0f),
                      glm::vec2(1.0f, 0.0f)};
      break;
    case SymbolShapeType::kWedge:
      rtn.flags = SymbolFlagBits::kTriangle;
      rtn.vertices = {glm::vec2(0.0f, 1.0f),
                      glm::vec2(-0.25f, -1.0f),
                      glm::vec2(0.25f, -1.0f),
                      glm::vec2(0.0f, 1.0f)};
      break;
    case SymbolShapeType::kArrow:
      rtn.flags = SymbolFlagBits::kXSymmetry | SymbolFlagBits::kUseWindingNumberTest;
      rtn.vertices = {glm::vec2(0.0f, 1.0f),
                      glm::vec2(0.4f, 0.1f),
                      glm::vec2(0.15f, 0.1f),
                      glm::vec2(0.15f, -1.0f),
                      glm::vec2(0.0f, -1.0f)};
      break;
    case SymbolShapeType::kAirplane:
      rtn.flags = SymbolFlagBits::kXSymmetry | SymbolFlagBits::kUseWindingNumberTest;
      rtn.vertices = {glm::vec2(0.0f, 1.0f),
                      glm::vec2(0.06f, 0.96f),
                      glm::vec2(0.08f, 0.9f),
                      glm::vec2(0.1f, 0.8f),
                      glm::vec2(0.1f, 0.2f),
                      glm::vec2(1.0f, -0.2f),  // wing tip
                      glm::vec2(1.0f, -0.36f),
                      glm::vec2(0.1f, -0.2f),
                      glm::vec2(0.1f, -0.7f),
                      glm::vec2(0.4f, -0.9f),  // stabilizer
                      glm::vec2(0.38f, -0.98f),
                      glm::vec2(0.0f, -0.92f)};
      break;
    case SymbolShapeType::kCOUNT:
      CHECK(false);
  }
  rtn.segment_count = rtn.vertices.size() - 1;
  return rtn;
}

void init_symbol_defs(SymbolDefinitions& symbol_defs) {
  if (!symbol_defs.initialized) {
    auto num_symbol_types = static_cast<uint32_t>(SymbolShapeType::kCOUNT);
    symbol_defs.resize(num_symbol_types);
    for (uint32_t type = 0; type < num_symbol_types; ++type) {
      auto def = get_symbol_def(static_cast<SymbolShapeType>(type));
      symbol_defs.flags[type] = def.flags;
      symbol_defs.segment_counts[type] = def.segment_count;
      symbol_defs.vert_offsets[type] = symbol_defs.verts.size();
      for (auto v : def.vertices) {
        symbol_defs.verts.push_back(v);
      }
    }
    symbol_defs.initialized = true;
  }
}

void generate_fast_symbol_interface_blocks(
    gfx::GlslStructBuilder& fragment_inputs,
    std::optional<gfx::GlslStructBuilder>& geometry_inputs,
    bool is_angle_uniform,
    bool has_accumulator,
    bool use_mesh_shader) {
  std::vector<gfx::GlslStructBuilder::Qualifier> quals = {
      gfx::GlslStructBuilder::Qualifier::kFlat};

  // flat uint64_t fRowId;
  // flat uint fShapeType;
  // flat vec4 fSymbolScale;
  // flat float fPointSize;
  // flat vec4 fFillColor;
  // flat vec4 fStrokeColor;
  // flat float fStrokeWidth;
  // flat float fApproxMaxCoverage;
  // flat float accumIdx (accumulation renders only)
  // vec2 UV (angle and geometry shader only)
  fragment_inputs.addMember("fRowId", gfx::BufferAttrType::kUint64, quals);
  fragment_inputs.addMember("fShapeType", gfx::BufferAttrType::kUint, quals);
  fragment_inputs.addMember("fSymbolScale", gfx::BufferAttrType::kVec4f, quals);
  fragment_inputs.addMember("fPointSize", gfx::BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fFillColor", gfx::BufferAttrType::kVec4f, quals);
  fragment_inputs.addMember("fStrokeColor", gfx::BufferAttrType::kVec4f, quals);
  fragment_inputs.addMember("fStrokeWidth", gfx::BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fApproxMaxCoverage", gfx::BufferAttrType::kFloat, quals);
  if (use_mesh_shader || geometry_inputs) {
    fragment_inputs.addMember("fUV", gfx::BufferAttrType::kVec2f);
  }
  if (has_accumulator) {
    fragment_inputs.addMember("accumIdx", gfx::BufferAttrType::kInt, quals);
  }

  // flat uint64_t gRowId
  // flat vec4 gFillColor
  // flat vec4 gStrokeColor
  // flat float gStrokeWidth
  // flat uint gShapeType
  // flat vec2 gPointSize
  // float float gAngle (only when not uniform)
  // flat int gAccumIdx (accumulation render only)
  if (geometry_inputs) {
    geometry_inputs->addMember("gRowId", gfx::BufferAttrType::kUint64, quals);
    geometry_inputs->addMember("gFillColor", gfx::BufferAttrType::kVec4f, quals);
    geometry_inputs->addMember("gStrokeColor", gfx::BufferAttrType::kVec4f, quals);
    geometry_inputs->addMember("gStrokeWidth", gfx::BufferAttrType::kFloat, quals);
    geometry_inputs->addMember("gShapeType", gfx::BufferAttrType::kUint, quals);
    geometry_inputs->addMember("gPointSize", gfx::BufferAttrType::kVec2f, quals);
    if (!is_angle_uniform) {
      geometry_inputs->addMember("gAngle", gfx::BufferAttrType::kFloat, quals);
    }
    if (has_accumulator) {
      geometry_inputs->addMember("gAccumIdx", gfx::BufferAttrType::kInt, quals);
    }
  }
}

}  // namespace QueryRenderer
