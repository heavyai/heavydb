/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(std430, binding = 0) uniform SHARED_UBO {
  mat4 viewTM;
};

// A vertex input needs no location of its own: it is matched against the reflection
in vec3 in_position;

layout(location = 0) out float out_depth;

// Deliberately without a location, which is what this test is for. glslang cannot match
// it to the fragment shader's in_colour, so the compile has to fail rather than assign
// one and hope.
out vec3 out_colour;

void main() {
  gl_Position = viewTM * vec4(in_position, 1.0);
  out_depth = in_position.z;
  out_colour = in_position * 0.5 + 0.5;
}
