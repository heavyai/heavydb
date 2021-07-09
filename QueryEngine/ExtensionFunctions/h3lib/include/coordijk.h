/*
 * Copyright 2016-2018 Uber Technologies, Inc.
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
/** @file coordijk.h
 * @brief   Header file for CoordIJK functions including conversion from lat/lon
 *
 * References two Vec2d cartesian coordinate systems:
 *
 *    1. gnomonic: face-centered polyhedral gnomonic projection space with
 *             traditional scaling and x-axes aligned with the face Class II
 *             i-axes.
 *
 *    2. hex2d: local face-centered coordinate system scaled a specific H3 grid
 *             resolution unit length and with x-axes aligned with the local
 *             i-axes
 */

#ifndef COORDIJK_H
#define COORDIJK_H

// #include "geoCoord.h"
#include "QueryEngine/ExtensionFunctions/h3lib/include/h3api.h"
#include "QueryEngine/ExtensionFunctions/h3lib/include/vec2d.h"

/** @struct CoordIJK
 * @brief IJK hexagon coordinates
 *
 * Each axis is spaced 120 degrees apart.
 */
#define I_INDEX 0
#define J_INDEX 1
#define K_INDEX 2
#define CoordIJK(variable_name) int variable_name[3]
#define CoordIJK_ptr(variable_name) int* variable_name
#define CoordIJK_clone(ijk) \
  { ijk[I_INDEX], ijk[J_INDEX], ijk[K_INDEX] }
#define CoordIJK_copy(dest_ijk, src_ijk) \
  dest_ijk[I_INDEX] = src_ijk[I_INDEX];  \
  dest_ijk[J_INDEX] = src_ijk[J_INDEX];  \
  dest_ijk[K_INDEX] = src_ijk[K_INDEX];
#define CoordIJKArray(variable_name, size) int variable_name[size][3]
// typedef struct {
//     int i;  ///< i component
//     int j;  ///< j component
//     int k;  ///< k component
// } CoordIJK;

/** @brief CoordIJK unit vectors corresponding to the 7 H3 digits.
 */
DEVICE static const CoordIJKArray(UNIT_VECS, 7) = {
    {0, 0, 0},  // direction 0
    {0, 0, 1},  // direction 1
    {0, 1, 0},  // direction 2
    {0, 1, 1},  // direction 3
    {1, 0, 0},  // direction 4
    {1, 0, 1},  // direction 5
    {1, 1, 0}   // direction 6
};

/** @brief H3 digit representing ijk+ axes direction.
 * Values will be within the lowest 3 bits of an integer.
 */
typedef enum {
  /** H3 digit in center */
  CENTER_DIGIT = 0,
  /** H3 digit in k-axes direction */
  K_AXES_DIGIT = 1,
  /** H3 digit in j-axes direction */
  J_AXES_DIGIT = 2,
  /** H3 digit in j == k direction */
  JK_AXES_DIGIT = J_AXES_DIGIT | K_AXES_DIGIT, /* 3 */
  /** H3 digit in i-axes direction */
  I_AXES_DIGIT = 4,
  /** H3 digit in i == k direction */
  IK_AXES_DIGIT = I_AXES_DIGIT | K_AXES_DIGIT, /* 5 */
  /** H3 digit in i == j direction */
  IJ_AXES_DIGIT = I_AXES_DIGIT | J_AXES_DIGIT, /* 6 */
  /** H3 digit in the invalid direction */
  INVALID_DIGIT = 7,
  /** Valid digits will be less than this value. Same value as INVALID_DIGIT.
   */
  NUM_DIGITS = INVALID_DIGIT
} Direction;

// Internal functions

EXTENSION_INLINE bool _setIJK(CoordIJK(ijk), int i, int j, int k);
EXTENSION_NOINLINE bool _hex2dToCoordIJK(const Vec2d(v), CoordIJK(h));
EXTENSION_INLINE bool _ijkToHex2d(const CoordIJK(h), Vec2d(v));
EXTENSION_INLINE int _ijkMatches(const CoordIJK(c1), const CoordIJK(c2));
EXTENSION_INLINE bool _ijkAdd(const CoordIJK(h1), const CoordIJK(h2), CoordIJK(sum));
EXTENSION_INLINE bool _ijkSub(const CoordIJK(h1), const CoordIJK(h2), CoordIJK(diff));
EXTENSION_INLINE bool _ijkScale(CoordIJK(c), int factor);
EXTENSION_NOINLINE bool _ijkNormalize(CoordIJK(c));
EXTENSION_NOINLINE int _unitIjkToDigit(const CoordIJK(ijk));
EXTENSION_NOINLINE bool _upAp7(CoordIJK(ijk));
EXTENSION_NOINLINE bool _upAp7r(CoordIJK(ijk));
EXTENSION_NOINLINE bool _downAp7(CoordIJK(ijk));
EXTENSION_NOINLINE bool _downAp7r(CoordIJK(ijk));
// void _downAp3(CoordIJK* ijk);
// void _downAp3r(CoordIJK* ijk);
EXTENSION_NOINLINE bool _neighbor(CoordIJK(ijk), int digit);
EXTENSION_NOINLINE bool _ijkRotate60ccw(CoordIJK(ijk));
EXTENSION_NOINLINE bool _ijkRotate60cw(CoordIJK(ijk));
EXTENSION_NOINLINE int _rotate60ccw(int digit);
EXTENSION_NOINLINE int _rotate60cw(int digit);
// int ijkDistance(const CoordIJK* a, const CoordIJK* b);
// void ijkToIj(const CoordIJK* ijk, CoordIJ* ij);
// void ijToIjk(const CoordIJ* ij, CoordIJK* ijk);
// void ijkToCube(CoordIJK* ijk);
// void cubeToIjk(CoordIJK* ijk);

#include "QueryEngine/ExtensionFunctions/h3lib/lib/coordijk.hpp"

#endif
