/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Shading/shadingCommon.glsl"
#include "Shading/projection.glsl"
#include "Shading/lighting.glsl"
#include "Shading/toneOperators.glsl"
//</includes>

layout(std430, binding = kMaterialPropsBinding) buffer MATERIAL_PROPS_SSBO {
  MaterialProperties materials[];
};

//
// Modified Blinn-Phong highlights over lambert diffuse
// Mask diffuse using specular
//
vec4 blinn_phong_diffuse(vec3 albedo, float glossiness, float spec_strength, vec3 P, vec3 N, vec3 V) {
  vec3 color = vec3(0);
  float spec_total = 0.0;
  uint32_t nlights = get_num_lights();
  vec3 L, lt_color;
  // Light loop
  for (uint32_t i = 0; i < nlights; i++) {
    eval_light(i, P, L, lt_color);
    // Compute specular energy
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), max(glossiness, 1.0) * 4.0) * spec_strength;
    // Sum specular component (used for masking opacity)
    spec_total += spec;

    // Mask diffuse using specular component to better conserve energy
    float n_dot_l = max(dot(N, L), 0.0) * max(1.0 - spec, 0.0);

    // Sum color
    color += albedo * n_dot_l * lt_color + spec * lt_color;
  }

  // Apply tone operator
  // TODO: make configurable
  color = reinhard_extended_luminance(color, 1.5);

  // Skip gamma for blinn-phong as it tends to wash out too much
  // color = max(pow(color, vec3(0.4545)), vec3(0));

  return vec4(color, spec_total);
}

//
// Cook-Torrance
//
const float PI = 3.141592653589793;

// fresnel reflective term
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
  return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Normal distribuion term - use GGX
float distribution_ggx(float N_dot_H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float denom = N_dot_H * N_dot_H * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

// Geometric self-shadowing term
float geometry_schlick_smith(float N_dot_L, float N_dot_V, float roughness) {
  float r = (roughness + 1.0);
  float k = (r * r) / 8.0;
  float GL = N_dot_L / (N_dot_L * (1.0 - k) + k);
  float GV = N_dot_V / (N_dot_V * (1.0 - k) + k);
  return GL * GV;
}

// Compute Cook-Torrance shading using provided inputs
// Evaluate the Fresnel term only, modifying color with 'metallic'
// This could be integrated into the `cook_torrance` function below
vec3 cook_torrance_reflectivity(vec3 albedo, float metallic, vec3 N, vec3 V) {
  return fresnel_schlick(max(dot(N, V), 0.0), mix(vec3(0.04), albedo, metallic));
}

vec4 cook_torrance(vec3 albedo, float roughness, float metallic, vec3 P, vec3 N, vec3 V) {
  // Compute diffuse albedo using 'metallic' input with min clamping
  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  float N_dot_V = max(dot(N, V), 0.0);

  // Sum surface color and specular density estimate (for opacity modulation)
  vec3 color = vec3(0);
  vec3 kS = vec3(0);
  uint32_t nlights = get_num_lights();
  vec3 L, lt_color;

  // Light loop
  for (uint32_t i = 0; i < nlights; i++) {
    eval_light(i, P, L, lt_color);
    vec3 H = normalize(V + L);
    float N_dot_L = max(dot(N, L), 0.0);
    float N_dot_H = max(dot(N, H), 0.0);

    // cook-torrance brdf
    float D = distribution_ggx(N_dot_H, roughness);
    float G = geometry_schlick_smith(N_dot_L, N_dot_V, roughness);
    vec3 F = fresnel_schlick(N_dot_V, F0);

    vec3 specular = (D * F * G) / (4.0 * N_dot_L * N_dot_V + 0.0001);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    color += (kD * albedo / PI + specular) * lt_color * N_dot_L;

    kS += length(specular);
  }

  // Apply tone operator
  // TODO: make this configurable
  color = reinhard_extended_luminance(color, 1.0);

  // Apply gamma-correction
  color = max(pow(color, vec3(0.4545)), vec3(0));

  return vec4(color, kS);
}

// All inputs are in world space
vec4 shade_point(const MaterialProperties material, vec3 P, vec3 N, vec3 V) {
  // Get material properties from SSBO
  // Properties are defined in terms of Cook-Torrance
  // To alias properties for Blinn-Phong:
  // - Use 'roughness' as glossiness (specular exponent)
  // - Use 'metallic' as specular strength (multiplier)

  // Compute the color
  // color.a = specular opacity and overrides general opacity
  vec4 color;
  if (material.shadingModel == SHADING_MODEL_CONSTANT) {
    color = vec4(material.albedo); // no specular
  } else if (material.shadingModel == SHADING_MODEL_BLINN_PHONG) {
    color = blinn_phong_diffuse(material.albedo.rgb, material.roughness, material.metallic, P, N, V);
  } else {
    color = cook_torrance(material.albedo.rgb, material.roughness, material.metallic, P, N, V);
  }

  // Mix returned specular opacity with base surface opacity
  float o = mix(material.opacity, 1.0, color.a);

  // Pre-multiply alpha and return
  color = vec4(color.rgb * o, o);
  return color;
}

// All inputs are in world space
vec4 shade_point(const MaterialProperties material, vec3 P, vec3 N, vec3 V, out vec3 reflectivity) {
  // Get material properties from SSBO
  // Properties are defined in terms of Cook-Torrance
  // To alias properties for Blinn-Phong:
  // - Use 'roughness' as glossiness (specular exponent)
  // - Use 'metallic' as specular strength (multiplier)

  // Compute the color
  // color.a = specular opacity and overrides general opacity
  vec4 color;
  if (material.shadingModel == SHADING_MODEL_CONSTANT) {
    color = vec4(material.albedo.rgb, 0.0); // no specular
    reflectivity = vec3(0);
  } else if (material.shadingModel == SHADING_MODEL_BLINN_PHONG) {
    color = blinn_phong_diffuse(material.albedo.rgb, material.roughness, material.metallic, P, N, V);
    reflectivity = vec3(0);
  } else {
    color = cook_torrance(material.albedo.rgb, material.roughness, material.metallic, P, N, V);
    reflectivity = cook_torrance_reflectivity(material.albedo.rgb, material.metallic, N, V);
  }

  // Mix returned specular opacity with base surface opacity
  float o = mix(material.opacity, 1.0, color.a);

  // Pre-multiply alpha and return
  color = vec4(color.rgb * o, o);
  return color;
}
