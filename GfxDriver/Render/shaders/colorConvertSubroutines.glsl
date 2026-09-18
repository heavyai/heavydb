/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Color-space conversion functions
//

// function signature
// vec4 transformColorToRGB(in vec4);

// pass-through
vec4 transformRGBtoRGB(in vec4 incolorRGB) {
  return incolorRGB;
}

// HSL -> RGB
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

const float Kn = 18,
              Xn = 0.950470,  // D65 standard referent
              Yn = 1, Zn = 1.088830,
              t0 = 4.0 / 29.0, t1 = 6.0 / 29.0,
              t2 = 3.0 * pow(6.0 / 29.0, 2.0),
              t3 = pow(6.0 / 29.0, 3.0);

float xyz2rgb(in float xyzchannel) {
    return (xyzchannel <= 0.0031308 ? 12.92 * xyzchannel : 1.055 * pow(xyzchannel, 1 / 2.4) - 0.055);
}

float lab2xyz(in float labchannel) {
    return (labchannel > t1 ? labchannel * labchannel * labchannel : t2 * (labchannel - t0));
}

// LAB -> RGB
vec4 transformLABtoRGB(in vec4 incolorLAB) {
    float y = (incolorLAB[0] + 16.0) / 116.0, x = isnan(incolorLAB[1]) ? y : y + incolorLAB[1] / 500.0,
          z = isnan(incolorLAB[2]) ? y : y - incolorLAB[2] / 200.0;
    y = Yn * lab2xyz(y);
    x = Xn * lab2xyz(x);
    z = Zn * lab2xyz(z);
    return vec4(xyz2rgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z),  // D65 -> sRGB
                xyz2rgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
                xyz2rgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z),
                incolorLAB.a);
}

// HCL -> RGB
vec4 transformHCLtoRGB(in vec4 incolorHCL) {
  float h = radians(incolorHCL[0]);
  return transformLABtoRGB(
      vec4(incolorHCL[2], cos(h) * incolorHCL[1], sin(h) * incolorHCL[1], incolorHCL[3]));
}

//
// Unpack color functions
//

// function signature
//vec4 unpackColorSubroutine(in uint);

// RGB
vec4 unpackRGBAColor(in uint incolorRGB) {
  vec4 color;
  color.r = float((incolorRGB >> 24) & 0xFF) / 255.0;
  color.g = float((incolorRGB >> 16) & 0xFF) / 255.0;
  color.b = float((incolorRGB >> 8) & 0xFF) / 255.0;
  color.a = float(incolorRGB & 0xFF) / 255.0;
  return color;
}

// LAB
vec4 unpackLABColor(in uint incolorLAB) {
  vec4 color;
  color.r = 100.0 * (float((incolorLAB >> 24) & 0xFF) / 255.0);
  color.g = 256.0 * (float((incolorLAB >> 16) & 0xFF) / 255.0) - 128.0;
  color.b = 256.0 * (float((incolorLAB >> 8) & 0xFF) / 255.0) - 128.0;
  color.a = float(incolorLAB & 0xFF) / 255.0;
  return color;
}

// BGR (useful for unpacking count images - eg fragment counts)
vec4 unpackBGRColor(in uint incolorBGR, in float alpha) {
  vec4 color;
  color.r = float(incolorBGR & 0xFFU) / 255;
  color.g = float((incolorBGR >> 8) & 0xFF) / 255;
  color.b = float((incolorBGR >> 16) & 0XFF) / 255;
  color.a = alpha;
  return color;
}
