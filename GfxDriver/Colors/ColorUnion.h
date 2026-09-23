/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Colors/ColorHCL.h"
#include "GfxDriver/Colors/ColorHSL.h"
#include "GfxDriver/Colors/ColorLAB.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Colors/Utils.h"
#include "GfxDriver/TypeGLSL.h"

namespace gfx {

class ColorUnion {
 public:
  union UnionColor {
    ColorRGBA rgba;
    ColorHSL hsl;
    ColorLAB lab;
    ColorHCL hcl;
    UnionColor() {}
    ~UnionColor() {}
  };

  ColorUnion() : _tag(ColorType::RGBA) { colorData.rgba.set(0.0f, 0.0f, 0.0f, 1.0f); }

  explicit ColorUnion(const std::string& colorStr);

  explicit ColorUnion(const float chan0,
                      const float chan1,
                      const float chan2,
                      const float opacity = 1.0,
                      const ColorType& type = ColorType::RGBA);

  explicit ColorUnion(const uint8_t c0,
                      const uint8_t c1,
                      const uint8_t c2,
                      const uint8_t opacity = 255,
                      const ColorType type = ColorType::RGBA);

  explicit ColorUnion(const std::array<uint8_t, 4>& color,
                      const ColorType type = ColorType::RGBA);
  explicit ColorUnion(const uint32_t packedColor, const ColorType type = ColorType::RGBA);
  ColorUnion(const ColorUnion& color);

  ~ColorUnion();

  ColorUnion& operator=(const ColorUnion& rhs);

  void set(const uint8_t c0,
           const uint8_t c1,
           const uint8_t c2,
           const uint8_t opacity = 255,
           const ColorType type = ColorType::RGBA);

  void set(const uint32_t packedColor, const ColorType type = ColorType::RGBA);

  void set(const float chan0,
           const float chan1,
           const float chan2,
           const float opacity = 1.0,
           const ColorType type = ColorType::RGBA);

  float operator[](unsigned int channel) const;

  void initFromCSSString(const std::string& colorStr);
  void initFromPackedUInt(const uint32_t packedVal,
                          const ColorType type = ColorType::RGBA);

  float opacity() const;

  bool operator==(const ColorUnion& other) const;
  bool operator!=(const ColorUnion& other) const { return !operator==(other); }
  operator std::string() const;

  ColorType getType() const { return _tag; }

  template <typename ColorT,
            typename std::enable_if<is_color<ColorT>::value>::type* = nullptr>
  ColorT get() const {
    ColorT color;
    switch (_tag) {
      case ColorType::RGBA:
        convertColor(colorData.rgba, color);
        break;
      case ColorType::HSL:
        convertColor(colorData.hsl, color);
        break;
      case ColorType::LAB:
        convertColor(colorData.lab, color);
        break;
      case ColorType::HCL:
        convertColor(colorData.hcl, color);
        break;
      case ColorType::INVALID:
        CHECK(_tag == ColorType::INVALID);
    }
    return color;
  }

  std::array<float, 4> getColorArray() const;
  const std::array<float, 4>& getColorArrayRef() const;
  uint32_t getPackedColor() const;

  void convertToType(const ColorType colorType);

  static bool isColorString(const std::string& colorStr) {
    return ::gfx::isColorString(colorStr);
  }
  static bool isValidPackedType(const ColorType colorType);
  static std::vector<std::string> getPackedColorPrefixes();

  bool isValidTypeGLSL(const ::gfx::TypeGLSLShPtr& type_glsl);

  static bool isPackedTypeGLSL(const ::gfx::TypeGLSLShPtr& type_glsl) {
    return ColorRGBA::isPackedTypeGLSL(type_glsl);
  }

  static ::gfx::TypeGLSL<float, 4> getTypeGLSL() { return ::gfx::TypeGLSL<float, 4>(); }
  static ::gfx::TypeGLSLShPtr getTypeGLSLPtr() {
    return ::gfx::TypeGLSLShPtr(new ::gfx::TypeGLSL<float, 4>());
  }

 private:
  ColorType _tag;

  UnionColor colorData;
};

}  // namespace gfx
