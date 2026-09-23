/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 1) out uint idA;
layout(location = 2) out uint idB;
layout(location = 3) out uint tableId;

uniform usampler2D id1ASampler;
uniform usampler2D id1BSampler;
uniform usampler2D id2Sampler;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  uint srcIdA = texelFetch(id1ASampler, coord, 0).r;
  uint srcIdB = texelFetch(id1BSampler, coord, 0).r;
  uint srcTableId = texelFetch(id2Sampler, coord, 0).r;

  if (srcIdA == 0 && srcIdB == 0) {
    // don't overwrite existing IDs with 0
    discard;
  }

  idA = srcIdA;
  idB = srcIdB;
  tableId = srcTableId;
}
