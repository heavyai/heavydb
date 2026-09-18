/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex>

#include "GfxDriver/Colors/BaseColor4f.h"

namespace gfx {

using Wrap0to360 = ColorValidators::AngleWrap0to360;
using PassThru = ColorValidators::PassThruValidation<float>;
using ClampL = ColorValidators::ClampValidation<float, 0, 100>;

class ColorHCL : public BaseColor4f<Wrap0to360, PassThru, ClampL> {
 public:
  enum class Channel { HUE = 0, CHROMA, LIGHTNESS, OPACITY };

  ColorHCL() : BaseColor4f() {}
  explicit ColorHCL(const std::string& colorStr) : BaseColor4f() {
    initFromCSSString(colorStr);
  }
  explicit ColorHCL(const float l,
                    const float a,
                    const float b,
                    const float opacity = 1.0)
      : BaseColor4f(l, a, b, opacity, validateH, validateC, validateL) {}
  ColorHCL(const ColorHCL& color) : BaseColor4f(color) {}
  ColorHCL(const ColorUnion& color) : BaseColor4f() { operator=(color); }

  ~ColorHCL() {}

  ColorHCL& operator=(const ColorHCL& rhs) {
    BaseColor4f::operator=(rhs);
    return *this;
  }

  ColorHCL& operator=(const ColorUnion& rhs);

  float operator[](unsigned int channel) const {
    return BaseColor4f::operator[](channel);
  }

  void set(const float c0, const float c1, const float c2, const float opacity = 1.0) {
    BaseColor4f::set(c0, c1, c2, opacity, validateH, validateC, validateL);
  }

  void setFromColor(const ColorUnion& color) { operator=(color); }

  float opacity() const { return BaseColor4f::opacity(); }

  void initFromCSSString(const std::string& colorStr);

  float h() const { return _colorArray[static_cast<int>(Channel::HUE)]; }
  float c() const { return _colorArray[static_cast<int>(Channel::CHROMA)]; }
  float l() const { return _colorArray[static_cast<int>(Channel::LIGHTNESS)]; }

  std::array<float, 4> getColorArray() const { return BaseColor4f::getColorArray(); }
  const std::array<float, 4>& getColorArrayRef() const {
    return BaseColor4f::getColorArrayRef();
  }

  operator std::string() const { return "hcl" + BaseColor4f::operator std::string(); }

  static bool isColorString(const std::string& colorStr);

  static const std::array<std::string, 2> funcNames;

 private:
  static const std::array<std::string, 2> argRegexStrs;
  static const Wrap0to360 validateH;
  static const PassThru validateC;
  static const ClampL validateL;

  void initFromFuncArgs(const std::smatch& argMatch);

  friend ColorInitializer;
};

}  // namespace gfx
