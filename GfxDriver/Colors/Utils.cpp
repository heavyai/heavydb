/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/Utils.h"

#include <cmath>

#include "GfxDriver/Colors/ColorHCL.h"
#include "GfxDriver/Colors/ColorHSL.h"
#include "GfxDriver/Colors/ColorInitializer.h"
#include "GfxDriver/Colors/ColorLAB.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Math/Constants.h"

namespace gfx {

bool isColorString(const std::string& str) {
  return ColorInitializer::isColorString(str);
}

ColorType getColorTypeFromColorString(const std::string& str) {
  return ColorInitializer::getColorTypeFromString(str);
}

ColorType getColorTypeFromColorPrefix(const std::string& colorPrefix) {
  std::smatch match;
  if (ColorInitializer::getFuncMatch<ColorRGBA>(colorPrefix, match, false)) {
    return ColorType::RGBA;
  } else if (ColorInitializer::getFuncMatch<ColorHSL>(colorPrefix, match, false)) {
    return ColorType::HSL;
  } else if (ColorInitializer::getFuncMatch<ColorLAB>(colorPrefix, match, false)) {
    return ColorType::LAB;
  } else if (ColorInitializer::getFuncMatch<ColorHCL>(colorPrefix, match, false)) {
    return ColorType::HCL;
  }

  THROW_RUNTIME_EX("The string \"" + colorPrefix + "\" is not a valid color prefix.");
}

std::vector<std::string> getAllColorPrefixes() {
  std::vector<std::string> colorPrefixes;
  colorPrefixes.insert(std::end(colorPrefixes),
                       std::begin(ColorRGBA::funcNames),
                       std::end(ColorRGBA::funcNames));
  colorPrefixes.insert(std::end(colorPrefixes),
                       std::begin(ColorHSL::funcNames),
                       std::end(ColorHSL::funcNames));
  colorPrefixes.insert(std::end(colorPrefixes),
                       std::begin(ColorLAB::funcNames),
                       std::end(ColorLAB::funcNames));
  colorPrefixes.insert(std::end(colorPrefixes),
                       std::begin(ColorHCL::funcNames),
                       std::end(ColorHCL::funcNames));

  return colorPrefixes;
}

static const float Xn = 0.950470,  // D65 standard referent
    Yn = 1, Zn = 1.088830, t0 = 4.0f / 29.0f, t1 = 6.0 / 29.0, t2 = 3 * t1 * t1,
                   t3 = t1 * t1 * t1;

static float rgb2xyz(const float rgbchannel) {
  return (rgbchannel <= 0.04045 ? rgbchannel / 12.92
                                : std::pow((rgbchannel + 0.055) / 1.055, 2.4));
}

static float xyz2rgb(const float xyzchannel) {
  return (xyzchannel <= 0.0031308 ? 12.92 * xyzchannel
                                  : 1.055 * std::pow(xyzchannel, 1 / 2.4) - 0.055);
}

static float xyz2lab(const float xyzchannel) {
  return (xyzchannel > t3 ? std::pow(xyzchannel, 1.0f / 3.0f) : xyzchannel / t2 + t0);
}

static float lab2xyz(const float labchannel) {
  return (labchannel > t1 ? labchannel * labchannel * labchannel
                          : t2 * (labchannel - t0));
}

void convertColor(const ColorRGBA& from, ColorHSL& to) {
  float r = from.r(), g = from.g(), b = from.b();
  float h, s, l;
  float cmin, cmax;

  std::tie(cmin, cmax) = std::minmax({r, g, b});
  auto delta = cmax - cmin;

  l = (cmax + cmin) / 2.0;

  if (cmax == 0.0) {
    h = 0.0;
    s = 0.0;
  } else {
    s = delta / (1 - std::abs(2 * l - 1));
    if (cmax == r) {
      h = std::fmod((g - b) / delta, 6.0f);
    } else if (cmax == g) {
      h = 2.0 + (b - r) / delta;
    } else {  // cmax == b
      h = 4.0 + (r - g) / delta;
    }

    h *= 60.0;
  }

  to.set(h, s, l, from.opacity());
}

void convertColor(const ColorRGBA& from, ColorLAB& to) {
  float b = rgb2xyz(from.r()), a = rgb2xyz(from.g()), l = rgb2xyz(from.b()),
        x = xyz2lab((0.4124564 * b + 0.3575761 * a + 0.1804375 * l) / Xn),
        y = xyz2lab((0.2126729 * b + 0.7151522 * a + 0.0721750 * l) / Yn),
        z = xyz2lab((0.0193339 * b + 0.1191920 * a + 0.9503041 * l) / Zn);
  to.set(116 * y - 16, 500 * (x - y), 200 * (y - z), from.opacity());
}

void convertColor(const ColorRGBA& from, ColorHCL& to) {
  ColorLAB lab;
  convertColor(from, lab);
  convertColor(lab, to);
}

void convertColor(const ColorHSL& from, ColorRGBA& to) {
  float h = std::fmod(from.h(), 360.0f), s = from.s(), l = from.l();
  if (h < 0) {
    h += 360.0;
  }
  float r(0), g(0), b(0);
  float C = s * (1 - std::abs(2 * l - 1));
  float X = C * (1 - std::abs(std::fmod(h / 60.0f, 2.0f) - 1));
  float m = l - C / 2.0f;

  int huecat = static_cast<int>(h / 60.0);
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

  to.set(r + m, g + m, b + m, from.opacity());
}
void convertColor(const ColorHSL& from, ColorLAB& to) {
  ColorRGBA rgb;
  convertColor(from, rgb);
  convertColor(rgb, to);
}
void convertColor(const ColorHSL& from, ColorHCL& to) {
  ColorLAB lab;
  convertColor(from, lab);
  convertColor(lab, to);
}

void convertColor(const ColorLAB& from, ColorRGBA& to) {
  float y = (from.l() + 16) / 116, x = std::isnan(from.a()) ? y : y + from.a() / 500,
        z = std::isnan(from.b()) ? y : y - from.b() / 200;
  y = Yn * lab2xyz(y);
  x = Xn * lab2xyz(x);
  z = Zn * lab2xyz(z);
  to.set(xyz2rgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z),  // D65 -> sRGB
         xyz2rgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
         xyz2rgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z),
         from.opacity());
}
void convertColor(const ColorLAB& from, ColorHSL& to) {
  ColorRGBA rgb;
  convertColor(from, rgb);
  convertColor(rgb, to);
}
void convertColor(const ColorLAB& from, ColorHCL& to) {
  auto a = from.a();
  auto b = from.b();
  float h = Math::rad2deg(atan2(b, a));
  to.set(h, std::sqrt(a * a + b * b), from.l(), from.opacity());
}

void convertColor(const ColorHCL& from, ColorRGBA& to) {
  ColorLAB lab;
  convertColor(from, lab);
  convertColor(lab, to);
}
void convertColor(const ColorHCL& from, ColorHSL& to) {
  ColorLAB lab;
  convertColor(from, lab);
  convertColor(lab, to);
}
void convertColor(const ColorHCL& from, ColorLAB& to) {
  float h = Math::deg2rad(from.h());
  to.set(from.l(), std::cos(h) * from.c(), std::sin(h) * from.c(), from.opacity());
}

}  // namespace gfx
