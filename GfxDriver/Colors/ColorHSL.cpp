/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/ColorHSL.h"

#include "GfxDriver/Colors/ColorInitializer.h"
#include "GfxDriver/Colors/ColorUnion.h"

namespace gfx {

static const std::string regexStrAllFloat = "([-+]?\\d*\\.?\\d*)";
static const std::string regexStr0to1 = "(0(?:\\.\\d*)?|1(?:\\.0*)?)";

const ColorValidators::AngleWrap0to360 ColorHSL::validateH =
    ColorValidators::AngleWrap0to360();
const ColorValidators::Clamp0to1f ColorHSL::validateSL = ColorValidators::Clamp0to1f();

const std::array<std::string, 2> ColorHSL::funcNames = {"hsl", "hsla"};
const std::array<std::string, 2> ColorHSL::argRegexStrs = {
    "\\s*" + regexStrAllFloat + "\\s*,\\s*" + regexStr0to1 + "\\s*,\\s*" + regexStr0to1 +
        "\\s*",
    "\\s*" + regexStrAllFloat + "\\s*,\\s*" + regexStr0to1 + "\\s*,\\s*" + regexStr0to1 +
        "\\s*,\\s*" + regexStr0to1 + "\\s*"};

ColorHSL& ColorHSL::operator=(const ColorUnion& rhs) {
  return operator=(rhs.get<ColorHSL>());
}

void ColorHSL::initFromCSSString(const std::string& colorStr) {
  ColorInitializer::initColorFromColorString(*this, colorStr);
}

bool ColorHSL::isColorString(const std::string& colorStr) {
  static const std::regex hslRegex(
      "^" + funcNames[0] + "\\s*\\(" + argRegexStrs[0] + "\\)\\s*$",
      std::regex_constants::icase);
  static const std::regex hslaRegex(
      "^" + funcNames[1] + "\\s*\\(" + argRegexStrs[1] + "\\)\\s*$",
      std::regex_constants::icase);
  return (std::regex_match(colorStr, hslRegex) || std::regex_match(colorStr, hslaRegex));
}

void ColorHSL::initFromFuncArgs(const std::smatch& argMatch) {
  CHECK(argMatch.size() == 4 || argMatch.size() == 5);

  float h, s, l, a = 1.0;

  h = std::stof(argMatch[1]);
  s = std::stof(argMatch[2]);
  l = std::stof(argMatch[3]);

  if (argMatch.size() == 5) {
    a = std::stof(argMatch[4]);
  }
  set(h, s, l, a);
}

}  // namespace gfx
