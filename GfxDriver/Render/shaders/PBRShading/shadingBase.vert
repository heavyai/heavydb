/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Shading/projection.glsl"
//</includes>

// Vertex attributes
in vec3 in_position; // position (local object space)
in vec3 in_normal;   // normal (local object space)

layout(location = 0) out vec3 out_object_pos; // interpolated object space position
layout(location = 1) out vec3 out_world_pos;  // interpolated world space position
layout(location = 2) out vec3 out_view_pos;   // interpolated view (projected camera) position
layout(location = 3) out vec3 out_normal;     // interpolated normal in world space (for shading)
layout(location = 4) flat out int out_instance_index; // gl_InstanceIndex to look up instance params in fragment shader

// SSBO containing instance positions (xyz) and uniform scale (w)
layout(std430 /*binding = 0*/) buffer INSTANCE_DATA_SSBO {
  vec4 instanceData[];
};

// Model matrix along with the normal matrix (inv transpose of model matrix)
layout(std430, binding = 1) uniform MODEL_MATRICES_UBO_TYPE {
  mat4 modelTM;
  mat4 normalTM;
};

// Compute and output vertex positions
void output_position() {
  // Get data for the instance being drawn
  vec4 data = instanceData[gl_InstanceIndex];

  // Compute world position
  vec4 world_pos = modelTM * vec4(in_position.x * data.w + data.x,
                                  in_position.y * data.w + data.y,
                                  in_position.z * data.w + data.z,
                                  1.0);

  // Compute view position
  vec4 view_pos = viewProjTM * world_pos;

  // Output to fragment shader
  out_object_pos = in_position;
  out_world_pos = world_pos.xyz;
  out_view_pos = view_pos.xyz;
  out_instance_index = gl_InstanceIndex;

  gl_Position = view_pos;
}

// Compute and output the world space normal for shading
void output_normal() {
  out_normal = vec3(normalTM * vec4(in_normal, 1)).xyz;
}
