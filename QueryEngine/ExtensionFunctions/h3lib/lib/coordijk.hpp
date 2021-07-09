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
/** @file coordijk.c
 * @brief   Hex IJK coordinate systems functions including conversions to/from
 * lat/lon.
 */

#include "QueryEngine/ExtensionFunctions/h3lib/include/coordijk.h"

// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

#include "QueryEngine/ExtensionFunctions/h3lib/include/constants.h"
// #include "geoCoord.h"
// #include "mathExtensions.h"

/**
 * Sets an IJK coordinate to the specified component values.
 *
 * @param ijk The IJK coordinate to set.
 * @param i The desired i component value.
 * @param j The desired j component value.
 * @param k The desired k component value.
 */
EXTENSION_INLINE bool _setIJK(CoordIJK(ijk), int i, int j, int k) {
  ijk[I_INDEX] = i;
  ijk[J_INDEX] = j;
  ijk[K_INDEX] = k;
  return true;
}

// /**
//  * Determine the containing hex in ijk+ coordinates for a 2D cartesian
//  * coordinate vector (from DGGRID).
//  *
//  * @param v The 2D cartesian coordinate vector.
//  * @param h The ijk+ coordinates of the containing hex.
//  */
EXTENSION_NOINLINE bool _hex2dToCoordIJK(const Vec2d(v), CoordIJK(h)) {
  double a1, a2;
  double x1, x2;
  int m1, m2;
  double r1, r2;

  // quantize into the ij system and then normalize
  h[K_INDEX] = 0;

  a1 = fabs(v[X_INDEX]);
  a2 = fabs(v[Y_INDEX]);

  // first do a reverse conversion
  x2 = a2 / M_SIN60;
  x1 = a1 + x2 / 2.0;

  // check if we have the center of a hex
  m1 = x1;
  m2 = x2;

  // otherwise round correctly
  r1 = x1 - m1;
  r2 = x2 - m2;

  if (r1 < 0.5) {
    if (r1 < 1.0 / 3.0) {
      if (r2 < (1.0 + r1) / 2.0) {
        h[I_INDEX] = m1;
        h[J_INDEX] = m2;
      } else {
        h[I_INDEX] = m1;
        h[J_INDEX] = m2 + 1;
      }
    } else {
      if (r2 < (1.0 - r1)) {
        h[J_INDEX] = m2;
      } else {
        h[J_INDEX] = m2 + 1;
      }

      if ((1.0 - r1) <= r2 && r2 < (2.0 * r1)) {
        h[I_INDEX] = m1 + 1;
      } else {
        h[I_INDEX] = m1;
      }
    }
  } else {
    if (r1 < 2.0 / 3.0) {
      if (r2 < (1.0 - r1)) {
        h[J_INDEX] = m2;
      } else {
        h[J_INDEX] = m2 + 1;
      }

      if ((2.0 * r1 - 1.0) < r2 && r2 < (1.0 - r1)) {
        h[I_INDEX] = m1;
      } else {
        h[I_INDEX] = m1 + 1;
      }
    } else {
      if (r2 < (r1 / 2.0)) {
        h[I_INDEX] = m1 + 1;
        h[J_INDEX] = m2;
      } else {
        h[I_INDEX] = m1 + 1;
        h[J_INDEX] = m2 + 1;
      }
    }
  }

  // now fold across the axes if necessary

  if (v[X_INDEX] < 0.0) {
    if ((h[J_INDEX] % 2) == 0)  // even
    {
      long long int axisi = h[J_INDEX] / 2;
      long long int diff = h[I_INDEX] - axisi;
      h[I_INDEX] = h[I_INDEX] - 2.0 * diff;
    } else {
      long long int axisi = (h[J_INDEX] + 1) / 2;
      long long int diff = h[I_INDEX] - axisi;
      h[I_INDEX] = h[I_INDEX] - (2.0 * diff + 1);
    }
  }

  if (v[Y_INDEX] < 0.0) {
    h[I_INDEX] = h[I_INDEX] - (2 * h[J_INDEX] + 1) / 2;
    h[J_INDEX] = -1 * h[J_INDEX];
  }

  _ijkNormalize(h);

  return true;
}

/**
 * Find the center point in 2D cartesian coordinates of a hex.
 *
 * @param h The ijk coordinates of the hex.
 * @param v The 2D cartesian coordinates of the hex center point.
 */
EXTENSION_INLINE bool _ijkToHex2d(const CoordIJK(h), Vec2d(v)) {
  int i = h[I_INDEX] - h[K_INDEX];
  int j = h[J_INDEX] - h[K_INDEX];

  v[X_INDEX] = i - 0.5 * j;
  v[Y_INDEX] = j * M_SQRT3_2;
  return true;
}

/**
 * Returns whether or not two ijk coordinates contain exactly the same
 * component values.
 *
 * @param c1 The first set of ijk coordinates.
 * @param c2 The second set of ijk coordinates.
 * @return 1 if the two addresses match, 0 if they do not.
 */
EXTENSION_INLINE int _ijkMatches(const CoordIJK(c1), const CoordIJK(c2)) {
  return (c1[I_INDEX] == c2[I_INDEX] && c1[J_INDEX] == c2[J_INDEX] &&
          c1[K_INDEX] == c2[K_INDEX]);
}

/**
 * Add two ijk coordinates.
 *
 * @param h1 The first set of ijk coordinates.
 * @param h2 The second set of ijk coordinates.
 * @param sum The sum of the two sets of ijk coordinates.
 */
EXTENSION_INLINE bool _ijkAdd(const CoordIJK(h1), const CoordIJK(h2), CoordIJK(sum)) {
  sum[I_INDEX] = h1[I_INDEX] + h2[I_INDEX];
  sum[J_INDEX] = h1[J_INDEX] + h2[J_INDEX];
  sum[K_INDEX] = h1[K_INDEX] + h2[K_INDEX];
  return true;
}

/**
 * Subtract two ijk coordinates.
 *
 * @param h1 The first set of ijk coordinates.
 * @param h2 The second set of ijk coordinates.
 * @param diff The difference of the two sets of ijk coordinates (h1 - h2).
 */
EXTENSION_INLINE bool _ijkSub(const CoordIJK(h1), const CoordIJK(h2), CoordIJK(diff)) {
  diff[I_INDEX] = h1[I_INDEX] - h2[I_INDEX];
  diff[J_INDEX] = h1[J_INDEX] - h2[J_INDEX];
  diff[K_INDEX] = h1[K_INDEX] - h2[K_INDEX];
  return true;
}

/**
 * Uniformly scale ijk coordinates by a scalar. Works in place.
 *
 * @param c The ijk coordinates to scale.
 * @param factor The scaling factor.
 */
EXTENSION_INLINE bool _ijkScale(CoordIJK(c), int factor) {
  c[I_INDEX] *= factor;
  c[J_INDEX] *= factor;
  c[K_INDEX] *= factor;
  return true;
}

/**
 * Normalizes ijk coordinates by setting the components to the smallest possible
 * values. Works in place.
 *
 * @param c The ijk coordinates to normalize.
 */
EXTENSION_NOINLINE bool _ijkNormalize(CoordIJK(c)) {
  // remove any negative values
  if (c[I_INDEX] < 0) {
    c[J_INDEX] -= c[I_INDEX];
    c[K_INDEX] -= c[I_INDEX];
    c[I_INDEX] = 0;
  }

  if (c[J_INDEX] < 0) {
    c[I_INDEX] -= c[J_INDEX];
    c[K_INDEX] -= c[J_INDEX];
    c[J_INDEX] = 0;
  }

  if (c[K_INDEX] < 0) {
    c[I_INDEX] -= c[K_INDEX];
    c[J_INDEX] -= c[K_INDEX];
    c[K_INDEX] = 0;
  }

  // remove the min value if needed
  int min = c[I_INDEX];
  if (c[J_INDEX] < min)
    min = c[J_INDEX];
  if (c[K_INDEX] < min)
    min = c[K_INDEX];
  if (min > 0) {
    c[I_INDEX] -= min;
    c[J_INDEX] -= min;
    c[K_INDEX] -= min;
  }

  return true;
}

/**
 * Determines the H3 digit corresponding to a unit vector in ijk coordinates.
 *
 * @param ijk The ijk coordinates; must be a unit vector.
 * @return The H3 digit (0-6) corresponding to the ijk unit vector, or
 * INVALID_DIGIT on failure.
 */
EXTENSION_NOINLINE int _unitIjkToDigit(const CoordIJK(ijk)) {
  CoordIJK(c) = CoordIJK_clone(ijk);
  _ijkNormalize(c);

  Direction digit = INVALID_DIGIT;
  for (int i = CENTER_DIGIT; i < NUM_DIGITS; i++) {
    if (_ijkMatches(c, UNIT_VECS[i])) {
      digit = Direction(i);
      break;
    }
  }

  return digit;
}

/**
 * Find the normalized ijk coordinates of the indexing parent of a cell in a
 * counter-clockwise aperture 7 grid. Works in place.
 *
 * @param ijk The ijk coordinates.
 */
EXTENSION_NOINLINE bool _upAp7(CoordIJK(ijk)) {
  // convert to CoordIJ
  int i = ijk[I_INDEX] - ijk[K_INDEX];
  int j = ijk[J_INDEX] - ijk[K_INDEX];

  ijk[I_INDEX] = (int)lround((3 * i - j) / 7.0);
  ijk[J_INDEX] = (int)lround((i + 2 * j) / 7.0);
  ijk[K_INDEX] = 0;
  _ijkNormalize(ijk);

  return true;
}

/**
 * Find the normalized ijk coordinates of the indexing parent of a cell in a
 * clockwise aperture 7 grid. Works in place.
 *
 * @param ijk The ijk coordinates.
 */
EXTENSION_NOINLINE bool _upAp7r(CoordIJK(ijk)) {
  // convert to CoordIJ
  int i = ijk[I_INDEX] - ijk[K_INDEX];
  int j = ijk[J_INDEX] - ijk[K_INDEX];

  ijk[I_INDEX] = (int)lround((2 * i + j) / 7.0);
  ijk[J_INDEX] = (int)lround((3 * j - i) / 7.0);
  ijk[K_INDEX] = 0;
  _ijkNormalize(ijk);
  return true;
}

/**
 * Find the normalized ijk coordinates of the hex centered on the indicated
 * hex at the next finer aperture 7 counter-clockwise resolution. Works in
 * place.
 *
 * @param ijk The ijk coordinates.
 */
EXTENSION_NOINLINE bool _downAp7(CoordIJK(ijk)) {
  // res r unit vectors in res r+1
  CoordIJK(iVec) = {3, 0, 1};
  CoordIJK(jVec) = {1, 3, 0};
  CoordIJK(kVec) = {0, 1, 3};

  _ijkScale(iVec, ijk[I_INDEX]);
  _ijkScale(jVec, ijk[J_INDEX]);
  _ijkScale(kVec, ijk[K_INDEX]);

  _ijkAdd(iVec, jVec, ijk);
  _ijkAdd(ijk, kVec, ijk);

  _ijkNormalize(ijk);
  return true;
}

/**
 * Find the normalized ijk coordinates of the hex centered on the indicated
 * hex at the next finer aperture 7 clockwise resolution. Works in place.
 *
 * @param ijk The ijk coordinates.
 */
EXTENSION_NOINLINE bool _downAp7r(CoordIJK(ijk)) {
  // res r unit vectors in res r+1
  CoordIJK(iVec) = {3, 1, 0};
  CoordIJK(jVec) = {0, 3, 1};
  CoordIJK(kVec) = {1, 0, 3};

  _ijkScale(iVec, ijk[I_INDEX]);
  _ijkScale(jVec, ijk[J_INDEX]);
  _ijkScale(kVec, ijk[K_INDEX]);

  _ijkAdd(iVec, jVec, ijk);
  _ijkAdd(ijk, kVec, ijk);

  _ijkNormalize(ijk);
  return true;
}

/**
 * Find the normalized ijk coordinates of the hex in the specified digit
 * direction from the specified ijk coordinates. Works in place.
 *
 * @param ijk The ijk coordinates.
 * @param digit The digit direction from the original ijk coordinates.
 */
EXTENSION_NOINLINE bool _neighbor(CoordIJK(ijk), int digit) {
  if (digit > CENTER_DIGIT && digit < NUM_DIGITS) {
    _ijkAdd(ijk, UNIT_VECS[digit], ijk);
    _ijkNormalize(ijk);
  }
  return true;
}

/**
 * Rotates ijk coordinates 60 degrees counter-clockwise. Works in place.
 *
 * @param ijk The ijk coordinates.
 */
EXTENSION_NOINLINE bool _ijkRotate60ccw(CoordIJK(ijk)) {
  // unit vector rotations
  CoordIJK(iVec) = {1, 1, 0};
  CoordIJK(jVec) = {0, 1, 1};
  CoordIJK(kVec) = {1, 0, 1};

  _ijkScale(iVec, ijk[I_INDEX]);
  _ijkScale(jVec, ijk[J_INDEX]);
  _ijkScale(kVec, ijk[K_INDEX]);

  _ijkAdd(iVec, jVec, ijk);
  _ijkAdd(ijk, kVec, ijk);

  _ijkNormalize(ijk);
  return true;
}

/**
 * Rotates ijk coordinates 60 degrees clockwise. Works in place.
 *
 * @param ijk The ijk coordinates.
 */
EXTENSION_NOINLINE bool _ijkRotate60cw(CoordIJK(ijk)) {
  // unit vector rotations
  CoordIJK(iVec) = {1, 0, 1};
  CoordIJK(jVec) = {1, 1, 0};
  CoordIJK(kVec) = {0, 1, 1};

  _ijkScale(iVec, ijk[I_INDEX]);
  _ijkScale(jVec, ijk[J_INDEX]);
  _ijkScale(kVec, ijk[K_INDEX]);

  _ijkAdd(iVec, jVec, ijk);
  _ijkAdd(ijk, kVec, ijk);

  _ijkNormalize(ijk);
  return true;
}

/**
 * Rotates indexing digit 60 degrees counter-clockwise. Returns result.
 *
 * @param digit Indexing digit (between 1 and 6 inclusive)
 */
EXTENSION_NOINLINE int _rotate60ccw(int digit) {
  switch (Direction(digit)) {
    case K_AXES_DIGIT:
      return IK_AXES_DIGIT;
    case IK_AXES_DIGIT:
      return I_AXES_DIGIT;
    case I_AXES_DIGIT:
      return IJ_AXES_DIGIT;
    case IJ_AXES_DIGIT:
      return J_AXES_DIGIT;
    case J_AXES_DIGIT:
      return JK_AXES_DIGIT;
    case JK_AXES_DIGIT:
      return K_AXES_DIGIT;
    default:
      return digit;
  }
}

/**
 * Rotates indexing digit 60 degrees clockwise. Returns result.
 *
 * @param digit Indexing digit (between 1 and 6 inclusive)
 */
EXTENSION_NOINLINE int _rotate60cw(int digit) {
  switch (Direction(digit)) {
    case K_AXES_DIGIT:
      return JK_AXES_DIGIT;
    case JK_AXES_DIGIT:
      return J_AXES_DIGIT;
    case J_AXES_DIGIT:
      return IJ_AXES_DIGIT;
    case IJ_AXES_DIGIT:
      return I_AXES_DIGIT;
    case I_AXES_DIGIT:
      return IK_AXES_DIGIT;
    case IK_AXES_DIGIT:
      return K_AXES_DIGIT;
    default:
      return digit;
  }
}

// /**
//  * Find the normalized ijk coordinates of the hex centered on the indicated
//  * hex at the next finer aperture 3 counter-clockwise resolution. Works in
//  * place.
//  *
//  * @param ijk The ijk coordinates.
//  */
// void _downAp3(CoordIJK* ijk) {
//     // res r unit vectors in res r+1
//     CoordIJK iVec = {2, 0, 1};
//     CoordIJK jVec = {1, 2, 0};
//     CoordIJK kVec = {0, 1, 2};

//     _ijkScale(&iVec, ijk->i);
//     _ijkScale(&jVec, ijk->j);
//     _ijkScale(&kVec, ijk->k);

//     _ijkAdd(&iVec, &jVec, ijk);
//     _ijkAdd(ijk, &kVec, ijk);

//     _ijkNormalize(ijk);
// }

// /**
//  * Find the normalized ijk coordinates of the hex centered on the indicated
//  * hex at the next finer aperture 3 clockwise resolution. Works in place.
//  *
//  * @param ijk The ijk coordinates.
//  */
// void _downAp3r(CoordIJK* ijk) {
//     // res r unit vectors in res r+1
//     CoordIJK iVec = {2, 1, 0};
//     CoordIJK jVec = {0, 2, 1};
//     CoordIJK kVec = {1, 0, 2};

//     _ijkScale(&iVec, ijk->i);
//     _ijkScale(&jVec, ijk->j);
//     _ijkScale(&kVec, ijk->k);

//     _ijkAdd(&iVec, &jVec, ijk);
//     _ijkAdd(ijk, &kVec, ijk);

//     _ijkNormalize(ijk);
// }

// /**
//  * Finds the distance between the two coordinates. Returns result.
//  *
//  * @param c1 The first set of ijk coordinates.
//  * @param c2 The second set of ijk coordinates.
//  */
// int ijkDistance(const CoordIJK* c1, const CoordIJK* c2) {
//     CoordIJK diff;
//     _ijkSub(c1, c2, &diff);
//     _ijkNormalize(&diff);
//     CoordIJK absDiff = {abs(diff.i), abs(diff.j), abs(diff.k)};
//     return MAX(absDiff.i, MAX(absDiff.j, absDiff.k));
// }

// /**
//  * Transforms coordinates from the IJK+ coordinate system to the IJ coordinate
//  * system.
//  *
//  * @param ijk The input IJK+ coordinates
//  * @param ij The output IJ coordinates
//  */
// void ijkToIj(const CoordIJK* ijk, CoordIJ* ij) {
//     ij->i = ijk->i - ijk->k;
//     ij->j = ijk->j - ijk->k;
// }

// /**
//  * Transforms coordinates from the IJ coordinate system to the IJK+ coordinate
//  * system.
//  *
//  * @param ij The input IJ coordinates
//  * @param ijk The output IJK+ coordinates
//  */
// void ijToIjk(const CoordIJ* ij, CoordIJK* ijk) {
//     ijk->i = ij->i;
//     ijk->j = ij->j;
//     ijk->k = 0;

//     _ijkNormalize(ijk);
// }

// /**
//  * Convert IJK coordinates to cube coordinates, in place
//  * @param ijk Coordinate to convert
//  */
// void ijkToCube(CoordIJK* ijk) {
//     ijk->i = -ijk->i + ijk->k;
//     ijk->j = ijk->j - ijk->k;
//     ijk->k = -ijk->i - ijk->j;
// }

// /**
//  * Convert cube coordinates to IJK coordinates, in place
//  * @param ijk Coordinate to convert
//  */
// void cubeToIjk(CoordIJK* ijk) {
//     ijk->i = -ijk->i;
//     ijk->k = 0;
//     _ijkNormalize(ijk);
// }
