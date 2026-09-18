/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FRAGMENT SHADER
//<extensions>
#extension GL_ARB_shader_draw_parameters : require
//</extensions>
// FRAGMENT SHADER
// Inputs

// flat uint64_t fRowId
// flat int accumIdx (accumulation renders only)
// vec2 fNormDistCoords
// flat vec4 fColor
// flat float fOpacity
// flat float fStrokeOpacity
in <FragmentShaderInputs>;

#define useSSBO <useSSBO>

void maybeDiscard() {
  // clip round joint, avoiding sqrt
  if (dot(fNormDistCoords, fNormDistCoords) > 1.0) {
    discard;
  }
}

vec4 getFragmentColorOrDiscard() {
  maybeDiscard();
  return vec4(fColor.rgb, fColor.a * fOpacity * fStrokeOpacity);
}
