/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define ARRAY_SIZE 4

layout(location = 0) out vec4 color;

uniform sampler2D sampler2d;
uniform sampler2DArray sampler2dArray;
uniform sampler2DArray arrayOfSampler2dArray[ARRAY_SIZE];

void main() {
  // dummy sampling code
  color = texelFetch(sampler2d, ivec2(gl_FragCoord.xy), 0);
  color = color + texelFetch(sampler2dArray, ivec3(gl_FragCoord.xy, 0), 0);
  color = color + texelFetch(arrayOfSampler2dArray[gl_SampleID], ivec3(gl_FragCoord.xy, 0), 0);
}
