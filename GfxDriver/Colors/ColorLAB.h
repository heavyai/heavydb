/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex>

#include "GfxDriver/Colors/BaseColor4f.h"

namespace gfx {

using ClampL = ColorValidators::ClampValidation<float, 0, 100>;
using ClampAB = ColorValidators::ClampValidation<float, -128, 128>;

namespace details {
struct ConvertL {
  float operator()(const uint8_t& val) const {
    return Math::clamp<float>(float(val) / 255.0f, 0.0, 1.0) * 100.0;
  }
  uint8_t inverse(const float val) const {
    return static_cast<uint8_t>((Math::clamp(val, 0.0f, 100.0f) / 100.0f) * 255.0);
  }
};

struct ConvertAB {
  float operator()(const uint8_t& val) const {
    return Math::clamp(float(val) / 255.0f, 0.0f, 1.0f) * 256.0f - 128.0f;
  }
  uint8_t inverse(const float val) const {
    return static_cast<uint8_t>(((Math::clamp(val, -128.0f, 128.0f) + 128.0f) / 256.0f) *
                                255.0);
  }
};
}  // namespace details

class ColorLAB : public BaseColorPacked4f<details::ConvertL,
                                          details::ConvertAB,
                                          details::ConvertAB,
                                          ClampL,
                                          ClampAB,
                                          ClampAB> {
 public:
  enum class Channel { LIGHTNESS = 0, A, B, OPACITY };

  ColorLAB() : BaseColorPacked4f() {}
  explicit ColorLAB(const std::string& colorStr) : BaseColorPacked4f() {
    initFromCSSString(colorStr);
  }
  explicit ColorLAB(const float l,
                    const float a,
                    const float b,
                    const float opacity = 1.0)
      : BaseColorPacked4f(l, a, b, opacity, clampL, clampAB, clampAB) {}
  explicit ColorLAB(const std::array<uint8_t, 4>& color)
      : BaseColorPacked4f(color,
                          convertL,
                          convertAB,
                          convertAB,
                          clampL,
                          clampAB,
                          clampAB) {}
  explicit ColorLAB(const uint32_t packedColor)
      : BaseColorPacked4f(packedColor,
                          convertL,
                          convertAB,
                          convertAB,
                          clampL,
                          clampAB,
                          clampAB) {}
  ColorLAB(const ColorLAB& color) : BaseColorPacked4f(color) {}
  ColorLAB(const ColorUnion& color) : BaseColorPacked4f() { operator=(color); }

  ~ColorLAB() {}

  ColorLAB& operator=(const ColorLAB& rhs) {
    BaseColorPacked4f::operator=(rhs);
    return *this;
  }

  ColorLAB& operator=(const ColorUnion& rhs);

  float operator[](unsigned int channel) const {
    return BaseColorPacked4f::operator[](channel);
  }

  void set(const float c0, const float c1, const float c2, const float opacity = 1.0) {
    BaseColorPacked4f::set(c0, c1, c2, opacity, clampL, clampAB, clampAB);
  }

  void setFromColor(const ColorUnion& color) { operator=(color); }

  void setFromPackedComponents(const uint8_t c0,
                               const uint8_t c1,
                               const uint8_t c2,
                               const uint8_t opacity = 255) {
    BaseColorPacked4f::setFromPackedComponents(
        c0, c1, c2, opacity, convertL, convertAB, convertAB, clampL, clampAB, clampAB);
  }

  void setFromPackedColor(const uint32_t packedColor) {
    BaseColorPacked4f::setFromPackedColor(
        packedColor, convertL, convertAB, convertAB, clampL, clampAB, clampAB);
  }

  float opacity() const { return BaseColorPacked4f::opacity(); }

  void initFromCSSString(const std::string& colorStr);

  float l() const { return _colorArray[static_cast<int>(Channel::LIGHTNESS)]; }
  float a() const { return _colorArray[static_cast<int>(Channel::A)]; }
  float b() const { return _colorArray[static_cast<int>(Channel::B)]; }

  std::array<float, 4> getColorArray() const {
    return BaseColorPacked4f::getColorArray();
  }
  const std::array<float, 4>& getColorArrayRef() const {
    return BaseColorPacked4f::getColorArrayRef();
  }

  uint32_t getPackedColor() const {
    return BaseColorPacked4f::getPackedColor(convertL, convertAB, convertAB);
  }

  operator std::string() const {
    return "lab" + BaseColorPacked4f::operator std::string();
  }

  static bool isColorString(const std::string& colorStr);

  static const std::array<std::string, 2> funcNames;

 private:
  static const std::array<std::string, 2> argRegexStrs;
  static const details::ConvertL convertL;
  static const details::ConvertAB convertAB;
  static const ClampL clampL;
  static const ClampAB clampAB;

  void initFromFuncArgs(const std::smatch& argMatch);

  friend ColorInitializer;
};

}  // namespace gfx
