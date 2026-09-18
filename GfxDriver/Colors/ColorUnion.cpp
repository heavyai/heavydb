/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ColorUnion.h"

#include "GfxDriver/Colors/Utils.h"

namespace gfx {

template <typename Rtn, typename... Args1, typename... Args2>
static Rtn baseColorMemberFuncCall(const ColorType type,
                                   ColorUnion::UnionColor& colorUnion,
                                   Rtn (ColorRGBA::*rgbafunc)(Args1...),
                                   Rtn (ColorHSL::*hslfunc)(Args1...),
                                   Rtn (ColorLAB::*labfunc)(Args1...),
                                   Rtn (ColorHCL::*hclfunc)(Args1...),
                                   Args2&&... args) {
  switch (type) {
    case ColorType::RGBA:
      return (colorUnion.rgba.*rgbafunc)(std::forward<Args2>(args)...);
    case ColorType::HSL:
      return (colorUnion.hsl.*hslfunc)(std::forward<Args2>(args)...);
    case ColorType::LAB:
      return (colorUnion.lab.*labfunc)(std::forward<Args2>(args)...);
    case ColorType::HCL:
      return (colorUnion.hcl.*hclfunc)(std::forward<Args2>(args)...);
    case ColorType::INVALID:
      CHECK(type == ColorType::INVALID);
    default:
      THROW_RUNTIME_EX("Color type " + ::gfx::to_string(type) +
                       " is not supported in a ColorUnion");
  }
}

template <typename Rtn, typename... Args1, typename... Args2>
static Rtn baseColorMemberFuncCallConst(const ColorType type,
                                        const ColorUnion::UnionColor& colorUnion,
                                        Rtn (ColorRGBA::*rgbafunc)(Args1...) const,
                                        Rtn (ColorHSL::*hslfunc)(Args1...) const,
                                        Rtn (ColorLAB::*labfunc)(Args1...) const,
                                        Rtn (ColorHCL::*hclfunc)(Args1...) const,
                                        Args2&&... args) {
  switch (type) {
    case ColorType::RGBA:
      return (colorUnion.rgba.*rgbafunc)(std::forward<Args2>(args)...);
    case ColorType::HSL:
      return (colorUnion.hsl.*hslfunc)(std::forward<Args2>(args)...);
    case ColorType::LAB:
      return (colorUnion.lab.*labfunc)(std::forward<Args2>(args)...);
    case ColorType::HCL:
      return (colorUnion.hcl.*hclfunc)(std::forward<Args2>(args)...);
    case ColorType::INVALID:
      CHECK(type == ColorType::INVALID);
    default:
      THROW_RUNTIME_EX("Color type " + ::gfx::to_string(type) +
                       " is not supported in a ColorUnion");
  }
}

template <typename Rtn, typename... Args1, typename... Args2>
static Rtn basePackedColorMemberFuncCall(const ColorType type,
                                         ColorUnion::UnionColor& colorUnion,
                                         Rtn (ColorRGBA::*rgbafunc)(Args1...),
                                         Rtn (ColorLAB::*labfunc)(Args1...),
                                         Args2&&... args) {
  switch (type) {
    case ColorType::RGBA:
      return (colorUnion.rgba.*rgbafunc)(std::forward<Args2>(args)...);
    case ColorType::LAB:
      return (colorUnion.lab.*labfunc)(std::forward<Args2>(args)...);
    default:
      THROW_RUNTIME_EX("Color type " + ::gfx::to_string(type) +
                       " is not a supported packed color in a ColorUnion");
  }
}

template <typename Rtn, typename... Args1, typename... Args2>
static Rtn basePackedColorMemberFuncCallConst(const ColorType type,
                                              const ColorUnion::UnionColor& colorUnion,
                                              Rtn (ColorRGBA::*rgbafunc)(Args1...) const,
                                              Rtn (ColorLAB::*labfunc)(Args1...) const,
                                              Args2&&... args) {
  switch (type) {
    case ColorType::RGBA:
      return (colorUnion.rgba.*rgbafunc)(std::forward<Args2>(args)...);
    case ColorType::LAB:
      return (colorUnion.lab.*labfunc)(std::forward<Args2>(args)...);
    default:
      THROW_RUNTIME_EX("Color type " + ::gfx::to_string(type) +
                       " is not a supported packed color in a ColorUnion");
  }
}

template <typename ColorT>
static void convertUnionColor(const ColorType type,
                              ColorUnion::UnionColor& colorUnion,
                              ColorT& color) {
  switch (type) {
    case ColorType::RGBA:
      convertColor(colorUnion.rgba, color);
      break;
    case ColorType::HSL:
      convertColor(colorUnion.hsl, color);
      break;
    case ColorType::LAB:
      convertColor(colorUnion.lab, color);
      break;
    case ColorType::HCL:
      convertColor(colorUnion.hcl, color);
      break;
    case ColorType::INVALID:
      CHECK(type == ColorType::INVALID);
  }
}

ColorUnion::ColorUnion(const std::string& colorStr)
    : _tag(getColorTypeFromColorString(colorStr)) {
  baseColorMemberFuncCall(_tag,
                          colorData,
                          &ColorRGBA::initFromCSSString,
                          &ColorHSL::initFromCSSString,
                          &ColorLAB::initFromCSSString,
                          &ColorHCL::initFromCSSString,
                          colorStr);
}

ColorUnion::ColorUnion(const float chan0,
                       const float chan1,
                       const float chan2,
                       const float opacity,
                       const ColorType& type)
    : _tag(type) {
  baseColorMemberFuncCall(_tag,
                          colorData,
                          &ColorRGBA::set,
                          &ColorHSL::set,
                          &ColorLAB::set,
                          &ColorHCL::set,
                          chan0,
                          chan1,
                          chan2,
                          opacity);
}

ColorUnion::ColorUnion(const uint8_t c0,
                       const uint8_t c1,
                       const uint8_t c2,
                       const uint8_t opacity,
                       const ColorType type)
    : ColorUnion() {
  set(c0, c1, c2, opacity, type);
}

ColorUnion::ColorUnion(const std::array<uint8_t, 4>& color, const ColorType type)
    : ColorUnion() {
  set(color[0], color[1], color[2], color[3], type);
}

ColorUnion::ColorUnion(const uint32_t packedColor, const ColorType type) : ColorUnion() {
  set(packedColor, type);
}

ColorUnion::ColorUnion(const ColorUnion& color) {
  operator=(color);
}

ColorUnion::~ColorUnion() {
  switch (_tag) {
    case ColorType::RGBA:
      colorData.rgba.~ColorRGBA();
      break;
    case ColorType::HSL:
      colorData.hsl.~ColorHSL();
      break;
    case ColorType::LAB:
      colorData.lab.~ColorLAB();
      break;
    case ColorType::HCL:
      colorData.hcl.~ColorHCL();
      break;
    case ColorType::INVALID:
      CHECK(_tag == ColorType::INVALID);
  }
}

ColorUnion& ColorUnion::operator=(const ColorUnion& rhs) {
  baseColorMemberFuncCall(rhs._tag,
                          colorData,
                          &ColorRGBA::setFromColor,
                          &ColorHSL::setFromColor,
                          &ColorLAB::setFromColor,
                          &ColorHCL::setFromColor,
                          rhs);

  _tag = rhs._tag;

  return *this;
}

void ColorUnion::set(const uint8_t c0,
                     const uint8_t c1,
                     const uint8_t c2,
                     const uint8_t opacity,
                     const ColorType type) {
  basePackedColorMemberFuncCall(type,
                                colorData,
                                &ColorRGBA::setFromPackedComponents,
                                &ColorLAB::setFromPackedComponents,
                                c0,
                                c1,
                                c2,
                                opacity);

  _tag = type;
}

void ColorUnion::set(const uint32_t packedColor, const ColorType type) {
  basePackedColorMemberFuncCall(type,
                                colorData,
                                &ColorRGBA::setFromPackedColor,
                                &ColorLAB::setFromPackedColor,
                                packedColor);

  _tag = type;
}

void ColorUnion::set(const float chan0,
                     const float chan1,
                     const float chan2,
                     const float opacity,
                     const ColorType type) {
  baseColorMemberFuncCall(type,
                          colorData,
                          &ColorRGBA::set,
                          &ColorHSL::set,
                          &ColorLAB::set,
                          &ColorHCL::set,
                          chan0,
                          chan1,
                          chan2,
                          opacity);

  _tag = type;
}

float ColorUnion::operator[](unsigned int channel) const {
  return baseColorMemberFuncCallConst(_tag,
                                      colorData,
                                      &ColorRGBA::operator[],
                                      &ColorHSL::operator[],
                                      &ColorLAB::operator[],
                                      &ColorHCL::operator[],
                                      channel);
}

void ColorUnion::initFromCSSString(const std::string& colorStr) {
  auto type = getColorTypeFromColorString(colorStr);
  baseColorMemberFuncCall(type,
                          colorData,
                          &ColorRGBA::initFromCSSString,
                          &ColorHSL::initFromCSSString,
                          &ColorLAB::initFromCSSString,
                          &ColorHCL::initFromCSSString,
                          colorStr);
  _tag = type;
}

void ColorUnion::initFromPackedUInt(const uint32_t packedVal, const ColorType type) {
  set(packedVal, type);
}

float ColorUnion::opacity() const {
  return baseColorMemberFuncCallConst(_tag,
                                      colorData,
                                      &ColorRGBA::opacity,
                                      &ColorHSL::opacity,
                                      &ColorLAB::opacity,
                                      &ColorHCL::opacity);
}

bool ColorUnion::operator==(const ColorUnion& other) const {
  if (_tag != other._tag) {
    return false;
  }
  return getColorArrayRef() == other.getColorArrayRef();
}

ColorUnion::operator std::string() const {
  return baseColorMemberFuncCallConst(_tag,
                                      colorData,
                                      &ColorRGBA::operator std::string,
                                      &ColorHSL::operator std::string,
                                      &ColorLAB::operator std::string,
                                      &ColorHCL::operator std::string);
}

std::array<float, 4> ColorUnion::getColorArray() const {
  return baseColorMemberFuncCallConst(_tag,
                                      colorData,
                                      &ColorRGBA::getColorArray,
                                      &ColorHSL::getColorArray,
                                      &ColorLAB::getColorArray,
                                      &ColorHCL::getColorArray);
}

const std::array<float, 4>& ColorUnion::getColorArrayRef() const {
  return baseColorMemberFuncCallConst(_tag,
                                      colorData,
                                      &ColorRGBA::getColorArrayRef,
                                      &ColorHSL::getColorArrayRef,
                                      &ColorLAB::getColorArrayRef,
                                      &ColorHCL::getColorArrayRef);
}

uint32_t ColorUnion::getPackedColor() const {
  return basePackedColorMemberFuncCallConst(
      _tag, colorData, &ColorRGBA::getPackedColor, &ColorLAB::getPackedColor);
}

void ColorUnion::convertToType(const ColorType colorType) {
  switch (colorType) {
    case ColorType::RGBA:
      convertUnionColor(_tag, colorData, colorData.rgba);
      break;
    case ColorType::HSL:
      convertUnionColor(_tag, colorData, colorData.hsl);
      break;
    case ColorType::LAB:
      convertUnionColor(_tag, colorData, colorData.lab);
      break;
    case ColorType::HCL:
      convertUnionColor(_tag, colorData, colorData.hcl);
      break;
    case ColorType::INVALID:
      CHECK(colorType == ColorType::INVALID);
  }
}

bool ColorUnion::isValidPackedType(const ColorType colorType) {
  if (colorType == ColorType::RGBA || colorType == ColorType::LAB) {
    return true;
  }

  return false;
}

std::vector<std::string> ColorUnion::getPackedColorPrefixes() {
  std::vector<std::string> colorPrefixes;
  colorPrefixes.insert(std::end(colorPrefixes),
                       std::begin(ColorRGBA::funcNames),
                       std::end(ColorRGBA::funcNames));
  colorPrefixes.insert(std::end(colorPrefixes),
                       std::begin(ColorLAB::funcNames),
                       std::end(ColorLAB::funcNames));
  return colorPrefixes;
}

bool ColorUnion::isValidTypeGLSL(const ::gfx::TypeGLSLShPtr& type_glsl) {
  switch (_tag) {
    case ColorType::RGBA:
      return ColorRGBA::isValidTypeGLSL(type_glsl);
    case ColorType::HSL:
      return ColorHSL::isValidTypeGLSL(type_glsl);
    case ColorType::LAB:
      return ColorLAB::isValidTypeGLSL(type_glsl);
    case ColorType::HCL:
      return ColorHCL::isValidTypeGLSL(type_glsl);
    case ColorType::INVALID:
      CHECK(_tag == ColorType::INVALID);
  }

  return false;
}

}  // namespace gfx
