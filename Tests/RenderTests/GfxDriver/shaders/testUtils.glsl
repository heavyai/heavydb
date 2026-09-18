/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

vec4 transformHSLtoRGB(in vec4 incolorHSL) {
  float h = mod(incolorHSL[0], 360.0), s = incolorHSL[1], l = incolorHSL[2];
  if (h < 0) {
    h += 360.0;
  }
  float r = 0, g = 0, b = 0;
  float C = s * (1 - abs(2 * l - 1));
  float X = C * (1 - abs(mod(h / 60.0, 2.0) - 1));
  float m = l - C / 2.0;

  int huecat = int(h / 60.0);
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

  return vec4(r + m, g + m, b + m, incolorHSL[3]);
}
