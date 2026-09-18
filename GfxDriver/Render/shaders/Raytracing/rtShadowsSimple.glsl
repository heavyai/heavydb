/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<extensions>
#extension GL_EXT_ray_tracing : require
//</extensions>

#define USE_RAYTRACED_SHADOWS 1

layout(location = 1) rayPayloadEXT bool is_shadowed;
uniform accelerationStructureEXT topLevelAS;

bool trace_shadow_ray(vec3 P, vec3 L, float tmin, float tmax) {
	is_shadowed = true; // start shadowed, miss shader will flip

	uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
	traceRayEXT(topLevelAS, ray_flags, 0xFF, 0, 0, 1, P, tmin, L, tmax, 1);
  return is_shadowed;
}
