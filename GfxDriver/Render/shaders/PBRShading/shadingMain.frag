/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Shading/shading.glsl"
//</includes>

// Minimal fragment shader supporting shading functions
// Outputs color directly to Framebuffer
//
// Fragment shader interface block declarations
// See mesh_Base.vert for position and normal output functions
// See mesh_Main_Shading.vert for minimal vertex shader main()
layout(location = 0) in vec3 fObjectPos;
layout(location = 1) in vec3 fWorldPos;
layout(location = 2) in vec3 fViewPos;
layout(location = 3) in vec3 fNormal;
layout(location = 4) flat in int fInstanceIndex;

// Framebuffer output
layout(location = 0) out vec4 out_color;

// Main entry point
void main() {
  // Determine albedo (surface color)
  // Use material property or derive from object space coords
  vec3 use_albedo = materials[fInstanceIndex].albedoSource == 0
                     ? materials[fInstanceIndex].albedo.rgb
                     : fObjectPos * 0.5 + 0.5;

  // Compute lit color (shade the point)
  vec3 V = projectionType == PROJECTION_TYPE_PERSPECTIVE
                             ? normalize(fWorldPos - cameraPosition.xyz)
                             : vec3(0, 0, 1);
  MaterialProperties material = materials[fInstanceIndex];
  vec4 color = shade_point(material, fWorldPos, normalize(fNormal), V, use_albedo);

  // Output to Framebuffer
  out_color = color;
}
