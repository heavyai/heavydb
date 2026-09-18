/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex>

#include "GfxDriver/Colors/BaseColor4f.h"

namespace gfx {

class ColorHSL : public BaseColor4f<ColorValidators::AngleWrap0to360,
                                    ColorValidators::Clamp0to1f,
                                    ColorValidators::Clamp0to1f> {
 public:
  enum class Channel { HUE = 0, SATURATION, LIGHTNESS, OPACITY };

  ColorHSL() : BaseColor4f() {}
  explicit ColorHSL(const std::string& colorStr) : BaseColor4f() {
    initFromCSSString(colorStr);
  }
  explicit ColorHSL(const float h,
                    const float s,
                    const float l,
                    const float opacity = 1.0)
      : BaseColor4f(h, s, l, opacity, validateH, validateSL, validateSL) {}
  ColorHSL(const ColorHSL& color) : BaseColor4f(color) {}
  ColorHSL(const ColorUnion& color) : BaseColor4f() { operator=(color); }

  ~ColorHSL() {}

  // TODO(croot): add copy constructors for the other color classes

  ColorHSL& operator=(const ColorHSL& rhs) {
    BaseColor4f::operator=(rhs);
    return *this;
  }

  ColorHSL& operator=(const ColorUnion& rhs);

  float operator[](unsigned int channel) const {
    return BaseColor4f::operator[](channel);
  }

  void set(const float c0, const float c1, const float c2, const float opacity = 1.0) {
    BaseColor4f::set(c0, c1, c2, opacity, validateH, validateSL, validateSL);
  }

  void setFromColor(const ColorUnion& color) { operator=(color); }

  float opacity() const { return BaseColor4f::opacity(); }

  // TODO(croot): add operator= overrides for the other color classes

  void initFromCSSString(const std::string& colorStr);

  float h() const { return _colorArray[static_cast<int>(Channel::HUE)]; }
  float s() const { return _colorArray[static_cast<int>(Channel::SATURATION)]; }
  float l() const { return _colorArray[static_cast<int>(Channel::LIGHTNESS)]; }

  std::array<float, 4> getColorArray() const { return BaseColor4f::getColorArray(); }
  const std::array<float, 4>& getColorArrayRef() const {
    return BaseColor4f::getColorArrayRef();
  }

  operator std::string() const { return "hsl" + BaseColor4f::operator std::string(); }

  static bool isColorString(const std::string& colorStr);

  static const std::array<std::string, 2> funcNames;

 private:
  static const std::array<std::string, 2> argRegexStrs;
  static const ColorValidators::AngleWrap0to360 validateH;
  static const ColorValidators::Clamp0to1f validateSL;

  void initFromFuncArgs(const std::smatch& argMatch);

  friend ColorInitializer;
};

}  // namespace gfx
