/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Utils/NoiseUtils.h"

#include <glm/gtc/noise.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace gfx {

const float F_PI = (3.1415926535897932384626433832795f);  //	pi
const float F_2PI = (6.283185307179586476925286766559f);  //	2*pi

//
// Fractal noises
//

// Standard fractal noise function using simplex noise as the basis
float fractal_noise(int iterations,
                    float x,
                    float y,
                    float scale,
                    float persistence,
                    float low,
                    float high) {
  float max_amplitude = 0.f;
  float amplitude = 1.0f;
  float frequency = scale;
  float noise = 0.f;

  for (int i = 0; i < iterations; ++i) {
    noise += glm::simplex<float>(glm::vec2(x * frequency, y * frequency)) * amplitude;
    max_amplitude += amplitude;
    amplitude *= persistence;
    frequency *= 2.0f;
  }

  noise /= max_amplitude;
  noise = noise * (high - low) * 0.5f + (high + low) * 0.5f;

  return noise;
}

//
// Cell (voronoi) noises
//

std::ostream& operator<<(std::ostream& os, const CellDistanceMetric& metric) {
  using e = CellDistanceMetric;
  switch (metric) {
    case e::kEuclidean:
      os << "Euclidean";
      break;
    case e::kEuclideanManhattan:
      os << "Euclidean Manhattan";
      break;
    case e::kEuclideanSquared:
      os << "Euclidean Squared";
      break;
    case e::kManhattan:
      os << "Manhattan";
      break;
    case e::kRadialManhattan:
      os << "Radial Manhattan";
      break;
  }
  return os;
}

std::string to_string(const CellDistanceMetric metric) {
  using e = CellDistanceMetric;
  switch (metric) {
    case e::kEuclidean:
      return "Euclidean";
    case e::kEuclideanManhattan:
      return "Euclidean Manhattan";
    case e::kEuclideanSquared:
      return "Euclidean Squared";
    case e::kManhattan:
      return "Manhattan";
    case e::kRadialManhattan:
      return "Radial Manhattan";
  }
  return "";
}

//	A hardwired lookup table to quickly determine how many feature
//	points should be in each spatial cube. We use a table so we don't
//	need to make multiple slower tests.  A random number indexed into
//	this array will give an approximate Poisson distribution of mean
//	density 2.5. Read the book for the longwinded explanation.
static int poisson_count[256] = {
    4, 3, 1, 1, 1, 2, 4, 2, 2, 2, 5, 1, 0, 2, 1, 2, 2, 0, 4, 3, 2, 1, 2, 1, 3, 2, 2, 4, 2,
    2, 5, 1, 2, 3, 2, 2, 2, 2, 2, 3, 2, 4, 2, 5, 3, 2, 2, 2, 5, 3, 3, 5, 2, 1, 3, 3, 4, 4,
    2, 3, 0, 4, 2, 2, 2, 1, 3, 2, 2, 2, 3, 3, 3, 1, 2, 0, 2, 1, 1, 2, 2, 2, 2, 5, 3, 2, 3,
    2, 3, 2, 2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 1, 3, 4, 2, 2, 2, 5, 4, 2, 4, 2, 2, 5, 4, 3, 2,
    2, 5, 4, 3, 3, 3, 5, 2, 2, 2, 2, 2, 3, 1, 1, 4, 2, 1, 3, 3, 4, 3, 2, 4, 3, 3, 3, 4, 5,
    1, 4, 2, 4, 3, 1, 2, 3, 5, 3, 2, 1, 3, 1, 3, 3, 3, 2, 3, 1, 5, 5, 4, 2, 2, 4, 1, 3, 4,
    1, 5, 3, 3, 5, 3, 4, 3, 2, 2, 1, 1, 1, 1, 1, 2, 4, 5, 4, 5, 4, 2, 1, 5, 1, 1, 2, 3, 3,
    3, 2, 5, 2, 3, 3, 2, 0, 2, 1, 1, 4, 2, 1, 3, 2, 1, 2, 2, 3, 2, 5, 5, 3, 4, 5, 5, 2, 4,
    4, 5, 3, 2, 2, 2, 1, 4, 2, 3, 3, 4, 2, 5, 4, 2, 4, 2, 2, 2, 4, 5, 3, 2};

//	This constant is manipulated to make sure that the mean value of F[0]
//	is 1.0. This makes an easy natural "scale" size of the cellular features.
#define DENSITY_ADJUSTMENT 0.398150f

//	the function to merge-sort a "cube" of samples into the current best-found
//	list of values.
static void add_samples(long xi,
                        long yi,
                        long zi,
                        long max_order,
                        CellDistanceMetric metric,
                        float at[3],
                        float* F,
                        glm::vec3* delta,
                        uint32_t* ID);

//-----------------------------------------------------------------------------------
// wCell_fbm_basic()
//-----------------------------------------------------------------------------------
float wCell_fbm_basic(const glm::vec3& p,
                      float octaves,
                      float H,
                      float lacunarity,
                      CellDistanceMetric metric) {
  float value, remainder;
  int i;
  glm::vec3 delta[1];
  float F[1];
  uint32_t ID[1];

  value = 0.0f;

  // inner loop of fractal construction
  glm::vec3 tp(p);
  float norm_weight = 1.0f;

  for (i = 0; i < octaves; ++i) {
    wCell(tp, 1, metric, F, delta, ID);
    value += F[0] * powf(lacunarity, -H * i);
    tp *= lacunarity;

    norm_weight += powf(lacunarity, -H * i);
  }

  remainder = octaves - (int)octaves;
  if (remainder) {  // add in octaves remainder
    wCell(tp, 1, metric, F, delta, ID);
    value += remainder * F[0] * powf(lacunarity, -H * i);
  }

  return value / norm_weight;
}

//-----------------------------------------------------------------------------------
// wCell_fbm_chips()
//-----------------------------------------------------------------------------------
float wCell_fbm_chips(const glm::vec3& p,
                      float octaves,
                      float H,
                      float lacunarity,
                      CellDistanceMetric metric) {
  float value, remainder;
  int i;
  glm::vec3 delta[2];
  float F[2];
  uint32_t ID[2];

  value = 0.0f;

  // inner loop of fractal construction
  glm::vec3 tp(p);
  float norm_weight = 1.0f;

  for (i = 0; i < octaves; ++i) {
    wCell(tp, 2, metric, F, delta, ID);
    value += (F[1] - F[0]) * powf(lacunarity, -H * i);
    tp *= lacunarity;

    norm_weight += powf(lacunarity, -H * i);
  }

  remainder = octaves - (int)octaves;
  if (remainder) {  // add in octaves remainder
    wCell(tp, 2, metric, F, delta, ID);
    value += remainder * (F[1] - F[0]) * powf(lacunarity, -H * i);
  }

  return value / norm_weight;
}

//-----------------------------------------------------------------------------------
// wCell_delta_fbm_basic()
//-----------------------------------------------------------------------------------
glm::vec3 wCell_delta_fbm_basic(const glm::vec3& p,
                                float octaves,
                                float H,
                                float lacunarity,
                                CellDistanceMetric metric) {
  float remainder;
  int i;
  glm::vec3 delta[1];
  float F[1];
  uint32_t ID[1];

  glm::vec3 value{};

  // inner loop of fractal construction
  glm::vec3 tp(p);
  float norm_weight = 1.0f;

  for (i = 0; i < octaves; ++i) {
    wCell(tp, 1, metric, F, delta, ID);
    value += delta[0] * powf(lacunarity, -H * i);
    tp *= lacunarity;

    norm_weight += powf(lacunarity, -H * i);
  }

  remainder = octaves - (int)octaves;
  if (remainder) {  // add in octaves remainder
    wCell(tp, 1, metric, F, delta, ID);
    value += remainder * delta[0] * powf(lacunarity, -H * i);
  }

  return value / norm_weight;
}

//-----------------------------------------------------------------------------------
// wCell_delta_fbm_chips()
//-----------------------------------------------------------------------------------
glm::vec3 wCell_delta_fbm_chips(const glm::vec3& p,
                                float octaves,
                                float H,
                                float lacunarity,
                                CellDistanceMetric metric) {
  float remainder;
  int i;
  glm::vec3 delta[2];
  float F[2];
  uint32_t ID[2];

  glm::vec3 value{};

  // inner loop of fractal construction
  glm::vec3 tp(p);
  float norm_weight = 1.0f;

  for (i = 0; i < octaves; ++i) {
    wCell(tp, 2, metric, F, delta, ID);
    value += (delta[1] - delta[0]) * powf(lacunarity, -H * i);
    tp *= lacunarity;

    norm_weight += powf(lacunarity, -H * i);
  }

  remainder = octaves - (int)octaves;
  if (remainder) {  // add in octaves remainder
    wCell(tp, 2, metric, F, delta, ID);
    value += remainder * (delta[1] - delta[0]) * powf(lacunarity, -H * i);
  }

  return value / norm_weight;
}

//-----------------------------------------------------------------------------------
// wCell()
// The main function
//-----------------------------------------------------------------------------------
void wCell(const glm::vec3& at,
           int max_order,
           CellDistanceMetric metric,
           float* F,
           glm::vec3* delta,
           uint32_t* ID) {
  float x2, y2, z2, mx2, my2, mz2;
  float new_at[3];
  int32_t int_at[3], i;

  // Initialize the F values to "huge" so they will be replaced by the
  // first real sample tests. Note we'll be storing and comparing the
  // SQUARED distance from the feature points to avoid lots of slow
  // sqrt() calls. We'll use sqrt() only on the final answer.
  for (i = 0; i < max_order; ++i) {
    F[i] = 999999.9f;
  }

  // Make our own local copy, multiplying to make mean(F[0])==1.0
  new_at[0] = DENSITY_ADJUSTMENT * at[0];
  new_at[1] = DENSITY_ADJUSTMENT * at[1];
  new_at[2] = DENSITY_ADJUSTMENT * at[2];

  // Find the integer cube holding the hit point
  int_at[0] = (int32_t)floorf(new_at[0]);  // A macro could make this slightly faster
  int_at[1] = (int32_t)floorf(new_at[1]);
  int_at[2] = (int32_t)floorf(new_at[2]);

  // A simple way to compute the closest neighbors would be to test all
  // boundary cubes exhaustively. This is simple with code like:
  //	long ii, jj, kk;
  //	for (ii=-1; ii<=1; ii++)
  //		for (jj=-1; jj<=1; jj++)
  //			for (kk=-1; kk<=1; kk++)
  //				add_samples(int_at[0]+ii,int_at[1]+jj,int_at[2]+kk,
  // max_order, new_at, F, delta, ID);
  // But this wastes a lot of time working on cubes which are known to be
  // too far away to matter! So we can use a more complex testing method
  // that avoids this needless testing of distant cubes. This doubles the
  // speed of the algorithm.

  // Test the central cube for closest point(s).
  add_samples(int_at[0], int_at[1], int_at[2], max_order, metric, new_at, F, delta, ID);

  // We test if neighbor cubes are even POSSIBLE contributors by examining the
  // combinations of the sum of the squared distances from the cube's lower
  // or upper corners.
  x2 = new_at[0] - int_at[0];
  y2 = new_at[1] - int_at[1];
  z2 = new_at[2] - int_at[2];

  mx2 = (1.0f - x2) * (1.0f - x2);
  my2 = (1.0f - y2) * (1.0f - y2);
  mz2 = (1.0f - z2) * (1.0f - z2);

  x2 *= x2;
  y2 *= y2;
  z2 *= z2;

  // Test 6 facing neighbors of center cube. These are closest and most
  // likely to have a close feature point.
  if (x2 < F[max_order - 1]) {
    add_samples(
        int_at[0] - 1, int_at[1], int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (y2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1] - 1, int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (z2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1], int_at[2] - 1, max_order, metric, new_at, F, delta, ID);
  }
  if (mx2 < F[max_order - 1]) {
    add_samples(
        int_at[0] + 1, int_at[1], int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (my2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1] + 1, int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (mz2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1], int_at[2] + 1, max_order, metric, new_at, F, delta, ID);
  }
  // Test 12 "edge cube" neighbors if necessary. They're next closest.
  if (x2 + y2 < F[max_order - 1]) {
    add_samples(
        int_at[0] - 1, int_at[1] - 1, int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (x2 + z2 < F[max_order - 1]) {
    add_samples(
        int_at[0] - 1, int_at[1], int_at[2] - 1, max_order, metric, new_at, F, delta, ID);
  }
  if (y2 + z2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1] - 1, int_at[2] - 1, max_order, metric, new_at, F, delta, ID);
  }
  if (mx2 + my2 < F[max_order - 1]) {
    add_samples(
        int_at[0] + 1, int_at[1] + 1, int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (mx2 + mz2 < F[max_order - 1]) {
    add_samples(
        int_at[0] + 1, int_at[1], int_at[2] + 1, max_order, metric, new_at, F, delta, ID);
  }
  if (my2 + mz2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1] + 1, int_at[2] + 1, max_order, metric, new_at, F, delta, ID);
  }
  if (x2 + my2 < F[max_order - 1]) {
    add_samples(
        int_at[0] - 1, int_at[1] + 1, int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (x2 + mz2 < F[max_order - 1]) {
    add_samples(
        int_at[0] - 1, int_at[1], int_at[2] + 1, max_order, metric, new_at, F, delta, ID);
  }
  if (y2 + mz2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1] - 1, int_at[2] + 1, max_order, metric, new_at, F, delta, ID);
  }
  if (mx2 + y2 < F[max_order - 1]) {
    add_samples(
        int_at[0] + 1, int_at[1] - 1, int_at[2], max_order, metric, new_at, F, delta, ID);
  }
  if (mx2 + z2 < F[max_order - 1]) {
    add_samples(
        int_at[0] + 1, int_at[1], int_at[2] - 1, max_order, metric, new_at, F, delta, ID);
  }
  if (my2 + z2 < F[max_order - 1]) {
    add_samples(
        int_at[0], int_at[1] + 1, int_at[2] - 1, max_order, metric, new_at, F, delta, ID);
  }
  // Final 8 "corner" cubes
  if (x2 + y2 + z2 < F[max_order - 1]) {
    add_samples(int_at[0] - 1,
                int_at[1] - 1,
                int_at[2] - 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (x2 + y2 + mz2 < F[max_order - 1]) {
    add_samples(int_at[0] - 1,
                int_at[1] - 1,
                int_at[2] + 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (x2 + my2 + z2 < F[max_order - 1]) {
    add_samples(int_at[0] - 1,
                int_at[1] + 1,
                int_at[2] - 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (x2 + my2 + mz2 < F[max_order - 1]) {
    add_samples(int_at[0] - 1,
                int_at[1] + 1,
                int_at[2] + 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (mx2 + y2 + z2 < F[max_order - 1]) {
    add_samples(int_at[0] + 1,
                int_at[1] - 1,
                int_at[2] - 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (mx2 + y2 + mz2 < F[max_order - 1]) {
    add_samples(int_at[0] + 1,
                int_at[1] - 1,
                int_at[2] + 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (mx2 + my2 + z2 < F[max_order - 1]) {
    add_samples(int_at[0] + 1,
                int_at[1] + 1,
                int_at[2] - 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }
  if (mx2 + my2 + mz2 < F[max_order - 1]) {
    add_samples(int_at[0] + 1,
                int_at[1] + 1,
                int_at[2] + 1,
                max_order,
                metric,
                new_at,
                F,
                delta,
                ID);
  }

  // We're done! Convert everything to right size scale
  for (i = 0; i < max_order; ++i) {
    F[i] = sqrtf(F[i]) * (1.0f / DENSITY_ADJUSTMENT);
    if (delta) {
      delta[i] = delta[i] * (1.0f / DENSITY_ADJUSTMENT);
    }
  }

  return;
}

//-----------------------------------------------------------------------------------
// add_samples()
//-----------------------------------------------------------------------------------
static void add_samples(long xi,
                        long yi,
                        long zi,
                        long max_order,
                        CellDistanceMetric metric,
                        float at[3],
                        float* F,
                        glm::vec3* delta,
                        uint32_t* ID) {
  float dx, dy, dz, fx, fy, fz, d2;
  int32_t count, i, j, index;
  uint32_t seed, this_id;

  // Each cube has a random number seed based on the cube's ID number.
  // The seed might be better if it were a nonlinear hash like Perlin uses
  // for noise but we do very well with this faster simple one.
  // Our LCG uses Knuth-approved constants for maximal periods.
  seed = 702395077 * xi + 915488749 * yi + 2120969693 * zi;

  // How many feature points are in this cube?
  uint32_t count_index = seed >> 24;
  count = poisson_count[count_index];  // 256 element lookup table. Use MSB

  seed = 1402024253 * seed + 586950981;  // churn the seed with good Knuth LCG

  for (j = 0; j < count; j++)  // test and insert each point into our solution
  {
    this_id = seed;
    seed = 1402024253 * seed + 586950981;  // churn

    // compute the 0..1 feature point location's XYZ
    fx = (seed + 0.5f) * (1.0f / 4294967296.0f);
    seed = 1402024253 * seed + 586950981;  // churn
    fy = (seed + 0.5f) * (1.0f / 4294967296.0f);
    seed = 1402024253 * seed + 586950981;  // churn
    fz = (seed + 0.5f) * (1.0f / 4294967296.0f);
    seed = 1402024253 * seed + 586950981;  // churn

    // delta from feature point to sample location
    dx = xi + fx - at[0];
    dy = yi + fy - at[1];
    dz = zi + fz - at[2];

    // Distance computation!  Lots of interesting variations are
    // possible here!
    // Biased "stretched"   A*dx*dx+B*dy*dy+C*dz*dz
    // Manhattan distance   fabs(dx)+fabs(dy)+fabs(dz)
    // Radial Manhattan:    A*fabs(dR)+B*fabs(dTheta)+C*dz
    // Superquadratic:      pow(fabs(dx), A) + pow(fabs(dy), B) + pow(fabs(dz),C)

    // Go ahead and make your own! Remember that you must insure that
    // new distance function causes large deltas in 3D space to map into
    // large deltas in your distance function, so our 3D search can find
    // them! [Alternatively, change the search algorithm for your special
    // cases.]

    switch (metric) {
      case CellDistanceMetric::kEuclidean:
        d2 = sqrtf(dx * dx + dy * dy + dz * dz);  // Euclidian distance
        break;
      case CellDistanceMetric::kEuclideanSquared:
        d2 = dx * dx + dy * dy + dz * dz;  // Euclidian distance, squared
        break;
      case CellDistanceMetric::kManhattan:
        d2 = sqrtf(fabsf(dx) + fabsf(dy) + fabsf(dz));  // Manhattan
        break;
      case CellDistanceMetric::kEuclideanManhattan:
        d2 = sqrtf(fabsf(dx) * fabsf(dx)) + (fabsf(dy) * fabsf(dy)) +
             (fabsf(dz) * fabsf(dz));  // euclidian Manhattan
        break;
      case CellDistanceMetric::kRadialManhattan: {
        //				float dR = sqrtf((dx*dx)+(dy*dy));
        //				float dTheta = (float)atan2(dx,dy);
        //				d2 = fabsf(dR) + fabsf(dTheta) + dz;

        float x1 = xi + fx;
        float y1 = yi + fy;
        float r1 = sqrtf((x1 * x1) + (y1 * y1));
        float r2 = sqrtf((at[0] * at[0]) + (at[1] * at[1]));
        float theta1 = ((float)atan2(y1, x1) + F_PI) / F_2PI;
        float theta2 = ((float)atan2(at[1], at[0]) + F_PI) / F_2PI;
        float dR = r1 - r2;
        float dTheta = theta1 - theta2;
        d2 = fabsf(dR) + fabsf(dTheta) + dz;
      } break;
      default:
        d2 = dx * dx + dy * dy + dz * dz;  // Euclidian distance, squared
    }

    if (d2 < F[max_order - 1])  // Is this point close enough to rememember?
    {
      // Insert the information into the output arrays if it's close enough.
      // We use an insertion sort.  No need for a binary search to find
      // the appropriate index.. usually we're dealing with order 2,3,4 so
      // we can just go through the list. If you were computing order 50
      // (wow!!) you could get a speedup with a binary search in the sorted
      // F[] list.

      index = max_order;
      while (index > 0 && d2 < F[index - 1]) {
        index--;
      }
      // We insert this new point into slot # <index>

      // Bump down more distant information to make room for this new point.
      for (i = max_order - 1; i-- > index;) {
        F[i + 1] = F[i];
        if (ID) {
          ID[i + 1] = ID[i];
        }
        if (delta) {
          delta[i + 1][0] = delta[i][0];
          delta[i + 1][1] = delta[i][1];
          delta[i + 1][2] = delta[i][2];
        }
      }
      // Insert the new point's information into the list.
      F[index] = d2;
      if (ID) {
        ID[index] = this_id;
      }
      if (delta) {
        delta[index][0] = dx;
        delta[index][1] = dy;
        delta[index][2] = dz;
      }
    }
  }

  return;
}

}  // namespace gfx
