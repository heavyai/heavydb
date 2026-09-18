/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

struct RayPayload {
  vec3 color;
  vec3 reflectivity;
  vec3 position;     // hit point in world space
  vec3 normal;       // surface normal in world space
  vec3 face_normal;  // geometric normal in world space
};
