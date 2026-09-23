/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#extension GL_ARB_fragment_shader_interlock : require

#include <DriverTests/testUtils.glsl>

layout(pixel_interlock_ordered) in;

layout(location = 0) in float in_key;

// both coherent and the interlock are required to guarantee memory ordering
layout (rgba8) coherent uniform image2D output_image;

vec4 blend_over(vec4 fore, vec4 back) {
	return fore + (1.0 - fore.a) * vec4(back.rgb * back.a, back.a);
}

void main(void) {
  beginInvocationInterlockARB();
  vec4 current_color = imageLoad(output_image, ivec2(gl_FragCoord.xy));

  // do a little work between load and store to increase likelyhood of sync issues
  const float alpha = 0.2;
  vec4 color = transformHSLtoRGB(vec4(in_key * 360.0, 1.0, 0.5, alpha));
  vec4 comped_color = blend_over(vec4(color.rgb, alpha), current_color);

  imageStore(output_image, ivec2(gl_FragCoord.xy), comped_color);
  endInvocationInterlockARB();
}
