/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// randomGenerators.glsl
//
// Functions for generating a pseudo-random 1d or 2d sample
// Most generators are low discrepancy sequences suitable for quasi monte carlo
// integration
//
// WARNING: These function can be slow, they are intended for experimental purposes
// For anything performance sensitive sequences should be precomputed on the host

//
// Random
//

float generate_random(vec2 co) {
  return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

//
// Weyl
//

// distribution visually similar to hammersley
vec2 generate_weyl(int i) {
  return fract(vec2(i * ivec2(12664745, 9560333)) /
               exp2(24.0));  // integer mul to avoid round-off
}

//
// Halton
//

// Halton sequence (general, requires prime basis)
float generate_halton(int prime_basis, int i) {
  float r = 0.0;
  float f = 1.0;
  while (i > 0) {
    f = f / float(prime_basis);
    r = r + f * float(i % prime_basis);
    i = int(floor(float(i) / float(prime_basis)));
  }
  return r;
}

// Halton sequence (basis 2)
float generate_halton_2(int i) {
  return float(bitfieldReverse(uint(i))) / 4294967296.0;
}

// Halton sequence (basis 2 and 3 for x and y)
vec2 generate_halton_23(int i) {
  return vec2(generate_halton_2(i), generate_halton(3, i));
}

// Halton sequence (basis 3 and 5 for x and y)
vec2 generate_halton_35(int i) {
  return vec2(generate_halton(3, i), generate_halton(5, i));
}

// Hamersley sequence (Halton basis 2 for x, y uniform distribution)
// Best convergence properties but requires fixed sample count
vec2 generate_hammersley(int i, int num_samples) {
  return vec2(generate_halton_2(i), float(i) / float(num_samples));
}
