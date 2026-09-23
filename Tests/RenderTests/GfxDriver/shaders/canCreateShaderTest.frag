/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 colorSample;

layout(push_constant) uniform SEPARATE_MS_PASS_PUSH_CONSTANTS {
  uint32_t sampleIndex;
} pushConstants;

uniform sampler2DMS colorTex;

void main(void)
{
  ivec2 coord = ivec2(gl_FragCoord.xy);
  colorSample = texelFetch(colorTex, coord, int(pushConstants.sampleIndex));
}
