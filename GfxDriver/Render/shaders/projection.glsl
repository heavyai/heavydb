/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Projection information
#define PROJECTION_TYPE_PERSPECTIVE   0
#define PROJECTION_TYPE_ORTHOGRAPHIC  1

// Projection information
layout(std430, binding = 2) uniform PROJECTION_UBO_TYPE {
  mat4 viewProjTM;      // Standard combined view + projection matrix
  mat4 viewInverse;
  mat4 projInverse;
  vec4 cameraPosition;  // Camera position (view point) in world space
  float nearClip;       // near clipping plane
  float farClip;        // far clipping place
  int projectionType;
};
