/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/ColorHCL.h"

#include "GfxDriver/Colors/ColorInitializer.h"
#include "GfxDriver/Colors/ColorUnion.h"

namespace gfx {

static const std::string regexStrAllFloat = "([-+]?\\d*\\.?\\d*)";
static const std::string regexStr0to1 = "(0(?:\\.\\d*)?|1(?:\\.0*)?)";

const Wrap0to360 ColorHCL::validateH = Wrap0to360();
const PassThru ColorHCL::validateC = PassThru();
const ClampL ColorHCL::validateL = ClampL();

const std::array<std::string, 2> ColorHCL::funcNames = {"hcl", "hcla"};
const std::array<std::string, 2> ColorHCL::argRegexStrs = {
    "\\s*" + regexStrAllFloat + "\\s*,\\s*" + regexStrAllFloat + "\\s*,\\s*" +
        regexStrAllFloat + "\\s*",
    "\\s*" + regexStrAllFloat + "\\s*,\\s*" + regexStrAllFloat + "\\s*,\\s*" +
        regexStrAllFloat + "\\s*,\\s*" + regexStr0to1 + "\\s*"};

ColorHCL& ColorHCL::operator=(const ColorUnion& rhs) {
  return operator=(rhs.get<ColorHCL>());
}

void ColorHCL::initFromCSSString(const std::string& colorStr) {
  ColorInitializer::initColorFromColorString(*this, colorStr);
}

bool ColorHCL::isColorString(const std::string& colorStr) {
  static const std::regex hclRegex(
      "^" + funcNames[0] + "\\s*\\(" + argRegexStrs[0] + "\\)\\s*$",
      std::regex_constants::icase);
  static const std::regex hclaRegex(
      "^" + funcNames[1] + "\\s*\\(" + argRegexStrs[1] + "\\)\\s*$",
      std::regex_constants::icase);
  return (std::regex_match(colorStr, hclRegex) || std::regex_match(colorStr, hclaRegex));
}

void ColorHCL::initFromFuncArgs(const std::smatch& argMatch) {
  CHECK(argMatch.size() == 4 || argMatch.size() == 5);

  float h, c, l, opacity = 1.0;

  h = std::stof(argMatch[1]);
  c = std::stof(argMatch[2]);
  l = std::stof(argMatch[3]);

  if (argMatch.size() == 5) {
    opacity = std::stof(argMatch[4]);
  }
  set(h, c, l, opacity);
}

}  // namespace gfx
