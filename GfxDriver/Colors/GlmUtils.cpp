/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/GlmUtils.h"

#include <cmath>

namespace gfx {

glm::vec4 HLSAtoRGBA(float h, float l, float s, float a) {
  h = fmod(h, 360.0f);
  if (h < 0) {
    h += 360.0;
  }
  float r = 0, g = 0, b = 0;
  float C = s * (1.0f - fabs(2.0f * l - 1.0f));
  float X = C * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
  float m = l - C / 2.0;

  int huecat = int(h / 60.0f);
  if (huecat == 0) {
    r = C;
    g = X;
  } else if (huecat == 1) {
    r = X;
    g = C;
  } else if (huecat == 2) {
    g = C;
    b = X;
  } else if (huecat == 3) {
    g = X;
    b = C;
  } else if (huecat == 4) {
    r = X;
    b = C;
  } else {
    r = C;
    b = X;
  }

  return glm::vec4(r + m, g + m, b + m, a);
}

}  // namespace gfx
