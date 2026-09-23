/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Material properties
#define SHADING_MODEL_CONSTANT      0
#define SHADING_MODEL_BLINN_PHONG   1
#define SHADING_MODEL_COOK_TORRANCE 2

struct MaterialProperties {
  vec4 albedo;
  float opacity;
  float roughness;   // glossiness
  float metallic;    // spec strength
  int shadingModel;
  int albedoSource;
  int pad[3];
};

const int kMaterialPropsBinding = 5;
const int kLightDataBinding = 6;
const int kLightUniformsBinding = 7;
