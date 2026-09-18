/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// lighting.glsl
// Defines uniform buffer for arrays of lights
//
// Functions:
// get_num_lights() to retrieve # of lights in the buffer
// eval_light() to get the light vector and rgb light energy for a specific light

#define POINT_LIGHT_TYPE      0
#define PARALLEL_LIGHT_TYPE   1

#define LIGHT_FLAG_DECAY  0x0001

struct Light {
  vec4 pos_or_dir; // alias for position or direction (world space)
  vec4 color;      // rgb = color, a = multiplier
  int type;        // light type (point or directional)
  uint32_t flags;  // flags (eg decay enable)
};

layout(std430, binding = kLightDataBinding) buffer LIGHT_DATA_SSBO {
  Light lights[];
} light_mgr;

// local uniforms (could be external)
layout(std430, binding = kLightUniformsBinding) uniform LIGHT_DATA_UNIFORMS {
  uint32_t numLights;
  float shadowBias;
};

uint32_t get_num_lights(){
  return numLights;
}

// Compute light energy color for a light in the SSBO
// light_index = light to lookup in light_mgr.lights (SSBO)
// P = point to be shaded (world space)
// L = return for light vector (world space)
// color = returned rgb energy at P
void eval_light(uint32_t light_index, vec3 P, out vec3 L, out vec3 color) {
  Light light = light_mgr.lights[light_index];
#if USE_RAYTRACED_SHADOWS == 1
  float tmax;
#endif
  if (light.type == POINT_LIGHT_TYPE) {
    L = normalize(light.pos_or_dir.xyz - P);
#if USE_RAYTRACED_SHADOWS == 1
    tmax = distance(L, P); // terminate ray at light
#endif
  } else if (light.type == PARALLEL_LIGHT_TYPE) {
    L = light.pos_or_dir.xyz;
#if USE_RAYTRACED_SHADOWS == 1
    tmax = 10000.0; // TODO (scb) this is arbitrary (should be infinite)
#endif
  } else {
    L = vec3(0);
    color = vec3(0);
    return;
  }

#if USE_RAYTRACED_SHADOWS == 1
  if (trace_shadow_ray(P, L, shadowBias, tmax)) {
    color = vec3(0);
    return;
  }
#endif

  // Apply decay (inverse square only)
  if ((light.flags & LIGHT_FLAG_DECAY) != 0) {
    float d = length(light.pos_or_dir.xyz - P);
    float atten = 1.0 / (d * d);
    color = light.color.rgb * atten * light.color.a;
  } else {
    color = light.color.rgb * light.color.a;
  }
}
