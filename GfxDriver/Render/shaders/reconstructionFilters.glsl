/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// reconstructionFilters.glsl
//
// Helper functions for generating image reconstruction filter weights given a pixel
// center offset
//
// WARNING: These function can be costly, they are intended for experimental purposes
// For anything performance sensitive samples and weights should be precomputed

//
// Mitchell-Netravali
//
// Good balance between blurring and ringing
const float kMitchell_B = 0.3333;  // blur amount
const float kMitchell_C = 0.3333;  // ringing amount
float eval_mitchell_filter(vec2 p) {
  float r = length(p);

  if (r >= 2.0) {
    return 0.0;
  }

  if (r < 1.0) {
    float D0 = 6.0 - 2.0 * kMitchell_B;                         // constant term
    float B0 = -18.0 + 12.0 * kMitchell_B + 6.0 * kMitchell_C;  // quadratic term
    float A0 = 12.0 - 9.0 * kMitchell_B - 6.0 * kMitchell_C;    // cubic term
    return r * (r * (A0 * r + B0)) + D0;
  } else {
    float D1 = 8.0 * kMitchell_B + 24.0 * kMitchell_C;    // constant term
    float C1 = -12.0 * kMitchell_B - 48.0 * kMitchell_C;  // linear term
    float B1 = 6.0 * kMitchell_B + 30.0 * kMitchell_C;    // quadratic term
    float A1 = -kMitchell_B - 6.0 * kMitchell_C;          // cubic term
    return r * (r * (A1 * r + B1) + C1) + D1;
  }
}
