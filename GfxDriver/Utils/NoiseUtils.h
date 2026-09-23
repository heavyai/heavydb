/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <ostream>

#include <glm/vec3.hpp>

namespace gfx {

//
// Fractal noises
//

// Standard fractal noise function using simplex noise as the basis
float fractal_noise(int iterations,
                    float x,
                    float y,
                    float scale,
                    float persistence = 0.5f,
                    float low = 0.0f,
                    float high = 1.0f);

//
// Cell (voronoi) noises
//
enum class CellDistanceMetric {
  kEuclidean,
  kEuclideanSquared,
  kManhattan,
  kEuclideanManhattan,
  kRadialManhattan
};

std::ostream& operator<<(std::ostream& os, const CellDistanceMetric& metric);
std::string to_string(const CellDistanceMetric metric);

//------------------------------------------------------------------------
//  wCell()
//
//  originally Worley(), the 'w' in wCell is in recognition of this function's origin
//
//  An implementation of the key cellular texturing basis
//  function. This function is hardwired to return an average F_1 value
//  of 1.0. It returns the <n> most closest feature point distances
//  F_1, F_2, .. F_n the vector delta to those points, and a 32 bit
//  seed for each of the feature points.  This function is not
//  difficult to extend to compute alternative information such as
//  higher order F values, to use the Manhattan distance metric, or
//  other fun perversions.
//
//  <at>    The input sample location.
//  <max_order>  Smaller values compute faster. < 5, read the book to extend it.
//  <metric> The distance metric to use
//  <F>     The output values of F_1, F_2, ..F[n] in F[0], F[1], F[n-1]
//  <delta> The output vector difference between the sample point and the n-th
//          closest feature point. Thus, the feature point's location is the
//          hit point minus this value. The DERIVATIVE of F is the unit
//          normalized version of this vector.
//	<ID>    The output 32 bit ID number which labels the feature point. This
//          is useful for domain partitions, especially for coloring flagstone
//          patterns.

//  This implementation is tuned for speed in a way that any order > 5
//  will likely have discontinuous artifacts in its computation of F5+.
//  This can be fixed by increasing the internal points-per-cube
//  density in the source code, at the expense of slower
//  computation. The book lists the details of this tuning.
//------------------------------------------------------------------------
void wCell(const glm::vec3& p,
           int max_order,
           CellDistanceMetric metric,
           float* value,
           glm::vec3* delta = nullptr,
           uint32_t* ID = nullptr);

// ------------------------------------------------------------------------------------
// fBm cell functions.
//
//    wCell_fbm_basic() returns a fBm version of F_1 order cells
//    wCell_fbm_chips() returns a fBm version of (F_2 - F_1) order cells
//
//    wCell_delta_fbm_basic() returns the delta of the F_1 order fBm cells.
//      useful for bump mapping
//    wCell_delta_fbm_chips() returns the delta of the (F_2 - F_1) order cells.
//      useful for bump mapping
// ------------------------------------------------------------------------------------
float wCell_fbm_basic(const glm::vec3& p,
                      float octaves,
                      float H,
                      float lacunarity,
                      CellDistanceMetric metric);
float wCell_fbm_chips(const glm::vec3& p,
                      float octaves,
                      float H,
                      float lacunarity,
                      CellDistanceMetric metric);

glm::vec3 wCell_delta_fbm_basic(const glm::vec3& p,
                                float octaves,
                                float H,
                                float lacunarity,
                                CellDistanceMetric metric);
glm::vec3 wCell_delta_fbm_chips(const glm::vec3& p,
                                float octaves,
                                float H,
                                float lacunarity,
                                CellDistanceMetric metric);

}  // namespace gfx
