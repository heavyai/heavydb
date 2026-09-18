/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/LineUtils.h"

using gfx::BufferAttrType;
using gfx::GlslStructBuilder;

namespace QueryRenderer {
void generate_line_interface_blocks(GlslStructBuilder& geometry_inputs,
                                    GlslStructBuilder& fragment_inputs,
                                    bool has_accumulator) {
  std::vector<GlslStructBuilder::Qualifier> quals = {GlslStructBuilder::Qualifier::kFlat};

  // Geometry shader inputs

  // uint64_t gRowId[]
  // int gAccumIdx[] (accumulation renders only)
  // float gOpacity[]
  // vec4 gColor[]
  // float gStrokeWidth[]
  // float gStrokeOpacity[]
  // int gLineJoin[]
  // float gMiterLimit[]
  geometry_inputs.addMember("gRowId", BufferAttrType::kUint64);
  geometry_inputs.addMember("gOpacity", BufferAttrType::kFloat);
  geometry_inputs.addMember("gColor", BufferAttrType::kVec4f);
  geometry_inputs.addMember("gStrokeWidth", BufferAttrType::kFloat);
  geometry_inputs.addMember("gStrokeOpacity", BufferAttrType::kFloat);
  geometry_inputs.addMember("gLineJoin", BufferAttrType::kInt);
  geometry_inputs.addMember("gMiterLimit", BufferAttrType::kFloat);
  if (has_accumulator) {
    geometry_inputs.addMember("gAccumIdx", BufferAttrType::kInt);
  }

  // Fragment shader inputs

  // flat uint64_t fRowId
  // flat int accumIdx (accumulation renders only)
  // vec2 fNormDistCoords
  // flat vec4 fColor
  // flat float fOpacity
  // flat float fStrokeOpacity
  fragment_inputs.addMember("fRowId", BufferAttrType::kUint64, quals);
  fragment_inputs.addMember("fNormDistCoords", BufferAttrType::kVec2f);
  fragment_inputs.addMember("fColor", BufferAttrType::kVec4f, quals);
  fragment_inputs.addMember("fOpacity", BufferAttrType::kFloat, quals);
  fragment_inputs.addMember("fStrokeOpacity", BufferAttrType::kFloat, quals);
  if (has_accumulator) {
    fragment_inputs.addMember("accumIdx", BufferAttrType::kInt, quals);
  }
}
}  // namespace QueryRenderer
