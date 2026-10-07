/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(std430, binding = 0) uniform SHARED_UBO {
  mat4 viewTM;
};

layout(location = 0) in float in_depth;

// The other half of the pair the vertex shader cannot be matched to
in vec3 in_colour;

// A fragment output needs no location of its own: it is matched against the reflection
out vec4 color;

void main() {
  color = vec4(in_colour * in_depth, 1.0);
}
