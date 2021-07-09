/*
 * Copyright 2016-2017 Uber Technologies, Inc.
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
/** @file vec2d.h
 * @brief   2D floating point vector functions.
 */

#ifndef VEC2D_H
#define VEC2D_H

#include "QueryEngine/ExtensionFunctions/h3lib/include/h3api.h"
// #include <stdbool.h>

#define X_INDEX 0
#define Y_INDEX 1
#define Vec2d(variable_name) double variable_name[2]
#define Vec2dArray(variable_name, size) double variable_name[size][2]
// /** @struct Vec2d
//  *  @brief 2D floating-point vector
//  */
// typedef struct {
//     double x;  ///< x component
//     double y;  ///< y component
// } Vec2d;

// Internal functions

EXTENSION_INLINE double _v2dMag(const Vec2d(v));
// void _v2dIntersect(const Vec2d* p0, const Vec2d* p1, const Vec2d* p2,
//                    const Vec2d* p3, Vec2d* inter);
// bool _v2dEquals(const Vec2d* p0, const Vec2d* p1);

#include "QueryEngine/ExtensionFunctions/h3lib/lib/vec2d.hpp"

#endif
