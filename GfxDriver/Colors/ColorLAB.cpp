/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/ColorLAB.h"

#include "GfxDriver/Colors/ColorInitializer.h"
#include "GfxDriver/Colors/ColorUnion.h"

namespace gfx {

static const std::string regexStrAllFloat = "([-+]?\\d*\\.?\\d*)";
static const std::string regexStr0to1 = "(0(?:\\.\\d*)?|1(?:\\.0*)?)";

const details::ConvertL ColorLAB::convertL = details::ConvertL();
const details::ConvertAB ColorLAB::convertAB = details::ConvertAB();
const ClampL ColorLAB::clampL = ClampL();
const ClampAB ColorLAB::clampAB = ClampAB();

const std::array<std::string, 2> ColorLAB::funcNames = {"lab", "laba"};
const std::array<std::string, 2> ColorLAB::argRegexStrs = {
    "\\s*" + regexStrAllFloat + "\\s*,\\s*" + regexStrAllFloat + "\\s*,\\s*" +
        regexStrAllFloat + "\\s*",
    "\\s*" + regexStrAllFloat + "\\s*,\\s*" + regexStrAllFloat + "\\s*,\\s*" +
        regexStrAllFloat + "\\s*,\\s*" + regexStr0to1 + "\\s*"};

ColorLAB& ColorLAB::operator=(const ColorUnion& rhs) {
  return operator=(rhs.get<ColorLAB>());
}

void ColorLAB::initFromCSSString(const std::string& colorStr) {
  ColorInitializer::initColorFromColorString(*this, colorStr);
}

bool ColorLAB::isColorString(const std::string& colorStr) {
  static const std::regex labRegex(
      "^" + funcNames[0] + "\\s*\\(" + argRegexStrs[0] + "\\)\\s*$",
      std::regex_constants::icase);
  static const std::regex labaRegex(
      "^" + funcNames[1] + "\\s*\\(" + argRegexStrs[1] + "\\)\\s*$",
      std::regex_constants::icase);
  return (std::regex_match(colorStr, labRegex) || std::regex_match(colorStr, labaRegex));
}

void ColorLAB::initFromFuncArgs(const std::smatch& argMatch) {
  CHECK(argMatch.size() == 4 || argMatch.size() == 5);

  float l, a, b, opacity = 1.0;

  l = std::stof(argMatch[1]);
  a = std::stof(argMatch[2]);
  b = std::stof(argMatch[3]);

  if (argMatch.size() == 5) {
    opacity = std::stof(argMatch[4]);
  }
  set(l, a, b, opacity);
}

}  // namespace gfx
