/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex>

#include "GfxDriver/Colors/BaseColor4f.h"

namespace gfx {

class ColorRGBA : public BaseColorPacked4f<PackedFloatColorConverters::ConvertUInt8To0to1,
                                           PackedFloatColorConverters::ConvertUInt8To0to1,
                                           PackedFloatColorConverters::ConvertUInt8To0to1,
                                           ColorValidators::Clamp0to1f,
                                           ColorValidators::Clamp0to1f,
                                           ColorValidators::Clamp0to1f> {
 public:
  enum class Channel { RED = 0, GREEN, BLUE, OPACITY };

  // static const std::unordered_map<std::string, std::string> colorKeywords;
  // static const std::regex rgbRegex;
  // static const std::regex rgbaRegex;
  // static const std::regex hexRegex;

  ColorRGBA() : BaseColorPacked4f() {}

  explicit ColorRGBA(const float r,
                     const float g,
                     const float b,
                     const float opacity = 1.0)
      : BaseColorPacked4f(r, g, b, opacity, clampRGB, clampRGB, clampRGB) {}
  explicit ColorRGBA(const std::array<float, 4>& color) : BaseColorPacked4f(color) {}
  explicit ColorRGBA(const std::string& colorStr) : BaseColorPacked4f() {
    initFromCSSString(colorStr);
  }
  explicit ColorRGBA(const std::array<uint8_t, 4>& color) : BaseColorPacked4f(color) {}
  explicit ColorRGBA(const uint32_t packedColor) : BaseColorPacked4f(packedColor) {}
  ColorRGBA(const ColorRGBA& color) : BaseColorPacked4f(color) {}
  ColorRGBA(const ColorUnion& color) : BaseColorPacked4f() { operator=(color); }

  // TODO(croot): add copy constructors for the other color classes

  ~ColorRGBA() {}

  ColorRGBA& operator=(const ColorRGBA& rhs) {
    BaseColorPacked4f::operator=(rhs);
    return *this;
  }

  ColorRGBA& operator=(const ColorUnion& rhs);

  float operator[](unsigned int channel) const {
    return BaseColorPacked4f::operator[](channel);
  }

  void set(const float c0, const float c1, const float c2, const float opacity = 1.0) {
    BaseColorPacked4f::set(c0, c1, c2, opacity, clampRGB, clampRGB, clampRGB);
  }

  void setFromColor(const ColorUnion& color) { operator=(color); }

  void setFromPackedComponents(const uint8_t c0,
                               const uint8_t c1,
                               const uint8_t c2,
                               const uint8_t opacity = 255) {
    BaseColorPacked4f::setFromPackedComponents(c0,
                                               c1,
                                               c2,
                                               opacity,
                                               convertRGB,
                                               convertRGB,
                                               convertRGB,
                                               clampRGB,
                                               clampRGB,
                                               clampRGB);
  }

  void setFromPackedColor(const uint32_t packedColor) {
    BaseColorPacked4f::setFromPackedColor(
        packedColor, convertRGB, convertRGB, convertRGB, clampRGB, clampRGB, clampRGB);
  }

  float opacity() const { return BaseColorPacked4f::opacity(); }

  // TODO(croot): add operator= overrides for the other color classes

  void initFromCSSString(const std::string& colorStr);

  float r() const { return _colorArray[static_cast<int>(Channel::RED)]; }
  float g() const { return _colorArray[static_cast<int>(Channel::GREEN)]; }
  float b() const { return _colorArray[static_cast<int>(Channel::BLUE)]; }
  float a() const { return _colorArray[static_cast<int>(Channel::OPACITY)]; }

  std::array<float, 4> getColorArray() const {
    return BaseColorPacked4f::getColorArray();
  }
  const std::array<float, 4>& getColorArrayRef() const {
    return BaseColorPacked4f::getColorArrayRef();
  }

  uint32_t getPackedColor() const {
    return BaseColorPacked4f::getPackedColor(convertRGB, convertRGB, convertRGB);
  }

  operator std::string() const;

  static bool isColorString(const std::string& colorStr);

  static const std::array<std::string, 2> funcNames;

 private:
  static const std::array<std::string, 2> argRegexStrs;
  static const PackedFloatColorConverters::ConvertUInt8To0to1 convertRGB;
  static const ColorValidators::Clamp0to1f clampRGB;

  static bool getNonFuncMatch(const std::string& colorStr, std::smatch& nonFuncMatch);
  void initFromFuncArgs(const std::smatch& argMatch);
  void initFromNonFunc(const std::string& colorStr, const std::smatch& nonFuncMatch);

  friend ColorInitializer;

  //  static const details::ConvertToRGBAChannel rgbaChannelConverter;
};

}  // namespace gfx
