/*
 * Copyright 2018 Uber Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/** @file vec3d.h
 * @brief   3D floating point vector functions.
 */

#ifndef VEC3D_H
#define VEC3D_H

// #include "geoCoord.h"

/** @struct Vec3D
 *  @brief 3D floating point structure
 */
#define Z_INDEX 2  // x/y index defined in vec2d.h
#define Vec3d(variable_name) double variable_name[3]
#define Vec3dArray(variable_name, size) double variable_name[size][3]

EXTENSION_NOINLINE bool _geoToVec3d(const GeoCoord(geo), Vec3d(v));
EXTENSION_NOINLINE double _pointSquareDist(const Vec3d(v1), const Vec3d(v2));

#include "QueryEngine/ExtensionFunctions/h3lib/lib/vec3d.hpp"

#endif
