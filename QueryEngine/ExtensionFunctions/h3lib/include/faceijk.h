/*
 * Copyright 2016-2019 Uber Technologies, Inc.
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
/** @file faceijk.h
 * @brief   FaceIJK functions including conversion to/from lat/lon.
 *
 *  References the Vec2d cartesian coordinate systems hex2d: local face-centered
 *     coordinate system scaled a specific H3 grid resolution unit length and
 *     with x-axes aligned with the local i-axes
 */

#ifndef FACEIJK_H
#define FACEIJK_H

#include "QueryEngine/ExtensionFunctions/h3lib/include/coordijk.h"
#include "QueryEngine/ExtensionFunctions/h3lib/include/h3api.h"
#include "QueryEngine/ExtensionFunctions/h3lib/include/vec2d.h"

/** @struct FaceIJK
 * @brief Face number and ijk coordinates on that face-centered coordinate
 * system
 */
#define FACE_INDEX 3  // i = 0, j = 1, k = 2, face = 3
#define FaceIJK(variable_name) int variable_name[4]
#define FaceIJK_clone(fijk) \
  { fijk[I_INDEX], fijk[J_INDEX], fijk[K_INDEX], fijk[FACE_INDEX] }
#define FaceIJK_copy(dest_ijk, src_ijk) \
  dest_ijk[I_INDEX] = src_ijk[I_INDEX]; \
  dest_ijk[J_INDEX] = src_ijk[J_INDEX]; \
  dest_ijk[K_INDEX] = src_ijk[K_INDEX]; \
  dest_ijk[FACE_INDEX] = src_ijk[FACE_INDEX];

// typedef struct {
//     int face;        ///< face number
//     CoordIJK coord;  ///< ijk coordinates on that face
// } FaceIJK;

/** @struct FaceOrientIJK
 * @brief Information to transform into an adjacent face IJK system
 */
#define CCWROT_60_INDEX 4  // i = 0, j = 1, k = 2, face = 3
#define FaceOrientIJK(variable_name) int variable_name[5]
#define FaceOrientIJK2DArray(variable_name, size1, size2) \
  int variable_name[size1][size2][5]
// typedef struct {
//     int face;            ///< face number
//     CoordIJK translate;  ///< res 0 translation relative to primary face
//     int ccwRot60;  ///< number of 60 degree ccw rotations relative to primary
//                    /// face
// } FaceOrientIJK;

// extern const GeoCoord faceCenterGeo[NUM_ICOSA_FACES];

// indexes for faceNeighbors table
/** IJ quadrant faceNeighbors table direction */
#define IJ 1
/** KI quadrant faceNeighbors table direction */
#define KI 2
/** JK quadrant faceNeighbors table direction */
#define JK 3

// /** Invalid face index */
// #define INVALID_FACE -1

/** Digit representing overage type */
typedef enum {
  /** No overage (on original face) */
  NO_OVERAGE = 0,
  /** On face edge (only occurs on substrate grids) */
  FACE_EDGE = 1,
  /** Overage on new face interior */
  NEW_FACE = 2
} Overage;

// Internal functions

EXTENSION_INLINE int isResClassIII(int res);
EXTENSION_NOINLINE bool _geoToFaceIjk(const GeoCoord(g), int res, FaceIJK(h));
EXTENSION_NOINLINE bool _geoToHex2d(const GeoCoord(g), int res, int* face, Vec2d(v));
EXTENSION_NOINLINE bool _faceIjkToGeo(const FaceIJK(h), int res, GeoCoord(g));
// void _faceIjkToGeoBoundary(const FaceIJK* h, int res, int start, int length,
//                            GeoBoundary* g);
// void _faceIjkPentToGeoBoundary(const FaceIJK* h, int res, int start, int length,
//                                GeoBoundary* g);
// void _faceIjkToVerts(FaceIJK* fijk, int* res, FaceIJK* fijkVerts);
// void _faceIjkPentToVerts(FaceIJK* fijk, int* res, FaceIJK* fijkVerts);
EXTENSION_NOINLINE bool _hex2dToGeo(const Vec2d(v),
                                    int face,
                                    int res,
                                    int substrate,
                                    GeoCoord(g));
EXTENSION_NOINLINE int _adjustOverageClassII(FaceIJK(fijk),
                                             int res,
                                             int pentLeading4,
                                             int substrate);
// Overage _adjustPentVertOverage(FaceIJK* fijk, int res);

#include "QueryEngine/ExtensionFunctions/h3lib/lib/faceijk.hpp"

#endif
