/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex>

#include "GfxDriver/Colors/Types.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/TypeGLSL.h"

namespace gfx {

// NOTE: I cannot make these base classes and their derived color classes polymorphic
// or add extra attributes indicating the type of the object as that would introduce
// extra bytes in their object size, and I need the colors
// to be able to be tightly packed against the float array that they all inherit
// from the base class.

// TODO(croot): consider making these classes polymorphic. If so, then some changes
// would be needed in QueryRenderer classes that pack these classes into vectors
// that are then uploaded for rendering.

class BaseOpacityValidator {
 protected:
  BaseOpacityValidator() {}
  ~BaseOpacityValidator() {}

  static const ColorValidators::Clamp0to1f opacityValidator;
};

template <class C0Validator = ColorValidators::PassThruValidation<float>,
          class C1Validator = ColorValidators::PassThruValidation<float>,
          class C2Validator = ColorValidators::PassThruValidation<float>>
class BaseColor4f : BaseOpacityValidator {
 public:
  static bool isValidTypeGLSL(const TypeGLSLShPtr& type_glsl,
                              bool floatArrayOnly = false) {
    CHECK(type_glsl);
    return type_glsl->attrType() == BufferAttrType::kVec4f;
  }

  static TypeGLSL<float, 4> getTypeGLSL() { return TypeGLSL<float, 4>(); }
  static TypeGLSLShPtr getTypeGLSLPtr() { return std::make_shared<TypeGLSL<float, 4>>(); }

  bool operator==(const BaseColor4f& rhs) const { return _colorArray == rhs._colorArray; }
  bool operator!=(const BaseColor4f& rhs) const { return !operator==(rhs); }

 protected:
  BaseColor4f() : _colorArray({{0, 0, 0, 1}}) {}
  explicit BaseColor4f(const float c0,
                       const float c1,
                       const float c2,
                       const float opacity = 1.0,
                       const C0Validator& c0Validate = C0Validator(),
                       const C1Validator& c1Validate = C1Validator(),
                       const C2Validator& c2Validate = C2Validator())
      : _colorArray({{c0Validate(c0),
                      c1Validate(c1),
                      c2Validate(c2),
                      opacityValidator(opacity)}}) {}
  explicit BaseColor4f(const std::array<float, 4>& color,
                       const C0Validator& c0Validate = C0Validator(),
                       const C1Validator& c1Validate = C1Validator(),
                       const C2Validator& c2Validate = C2Validator())
      : BaseColor4f(color[0],
                    color[1],
                    color[2],
                    color[3],
                    c0Validate,
                    c1Validate,
                    c2Validate) {}

  BaseColor4f(const BaseColor4f& color) : _colorArray(color._colorArray) {}
  BaseColor4f(const BaseColor4f&& color) : _colorArray(std::move(color._colorArray)) {}
  ~BaseColor4f() {}

  BaseColor4f& operator=(const BaseColor4f& rhs) {
    _colorArray = rhs._colorArray;
    return *this;
  }

  void set(const float c0,
           const float c1,
           const float c2,
           const float opacity = 1.0,
           const C0Validator& c0Validate = C0Validator(),
           const C1Validator& c1Validate = C1Validator(),
           const C2Validator& c2Validate = C2Validator()) {
    _colorArray[0] = c0Validate(c0);
    _colorArray[1] = c1Validate(c1);
    _colorArray[2] = c2Validate(c2);
    _colorArray[3] = opacityValidator(opacity);
  }

  void setFromColor(const ColorUnion& color) {
    THROW_RUNTIME_EX("TO BE OVERRIDDEN BY DERIVED CLASS");
  }

  float operator[](unsigned int channel) const { return _colorArray[channel]; }

  void initFromCSSString(const std::string& colorStr) {
    THROW_RUNTIME_EX("TO BE OVERRIDDEN BY DERIVED CLASS");
  }

  float opacity() const { return _colorArray[3]; }

  std::array<float, 4> getColorArray() const { return _colorArray; }
  const std::array<float, 4>& getColorArrayRef() const { return _colorArray; }

  static bool isColorString(const std::string& colorStr) {
    THROW_RUNTIME_EX("TO BE OVERRIDDEN BY DERIVED CLASS");
    return false;
  }

  operator std::string() const {
    std::ostringstream s;
    int i = 0;
    s << "(";
    for (i = 0; i < static_cast<int>(_colorArray.size()) - 1; ++i) {
      s << _colorArray[i] << ",";
    }
    s << _colorArray[i];

    s << ")";
    return s.str();
  }

  ColorArray _colorArray;

 private:
  // static const std::array<> funcNames
  // static const std::array<> argRegexStrs

  static bool getNonFuncMatch(const std::string& colorStr, std::smatch& nonFuncMatch) {
    return false;
  }

  void initFromFuncArgs(const std::smatch& argMatch) {
    THROW_RUNTIME_EX("NEEDS TO BE OVERRIDDEN BY DERIVED CLASS");
  }
  void initFromNonFunc(const std::string& colorStr, const std::smatch& nonFuncMatch) {
    // TODO(croot): Give a more detailed reason as to why
    THROW_RUNTIME_EX("Cannot initialize a Color object from the string \"" + colorStr +
                     "\"");
  }

  friend ColorInitializer;
};

class BaseOpacityConvertToFloat {
 protected:
  BaseOpacityConvertToFloat() {}
  ~BaseOpacityConvertToFloat() {}

  static const PackedFloatColorConverters::ConvertUInt8To0to1
      opacityConvertToFloatChannel;
};

template <class C0ConvertToFloatChannel,
          class C1ConvertToFloatChannel,
          class C2ConvertToFloatChannel,
          class C0Validator = ColorValidators::PassThruValidation<float>,
          class C1Validator = ColorValidators::PassThruValidation<float>,
          class C2Validator = ColorValidators::PassThruValidation<float>>
class BaseColorPacked4f : public BaseOpacityConvertToFloat,
                          public BaseColor4f<C0Validator, C1Validator, C2Validator> {
 public:
  static bool isValidTypeGLSL(const TypeGLSLShPtr& type_glsl) {
    CHECK(type_glsl);
    auto attr_type = type_glsl->attrType();
    return attr_type == BufferAttrType::kVec4f || attr_type == BufferAttrType::kInt ||
           attr_type == BufferAttrType::kUint || attr_type == BufferAttrType::kInt64 ||
           attr_type == BufferAttrType::kUint64;
  }

  static bool isPackedTypeGLSL(const TypeGLSLShPtr& type_glsl) {
    CHECK(type_glsl);
    auto attr_type = type_glsl->attrType();
    return attr_type == BufferAttrType::kInt || attr_type == BufferAttrType::kUint ||
           attr_type == BufferAttrType::kInt64 || attr_type == BufferAttrType::kUint64;
  }

 protected:
  BaseColorPacked4f() : BaseColor4f<C0Validator, C1Validator, C2Validator>() {}
  explicit BaseColorPacked4f(const float c0,
                             const float c1,
                             const float c2,
                             const float opacity = 1.0,
                             const C0Validator& c0Validate = C0Validator(),
                             const C1Validator& c1Validate = C1Validator(),
                             const C2Validator& c2Validate = C2Validator())
      : BaseColor4f<C0Validator, C1Validator, C2Validator>(c0,
                                                           c1,
                                                           c2,
                                                           opacity,
                                                           c0Validate,
                                                           c1Validate,
                                                           c2Validate) {}
  explicit BaseColorPacked4f(const std::array<float, 4>& color,
                             const C0Validator& c0Validate = C0Validator(),
                             const C1Validator& c1Validate = C1Validator(),
                             const C2Validator& c2Validate = C2Validator())
      : BaseColor4f<C0Validator, C1Validator, C2Validator>(color,
                                                           c0Validate,
                                                           c1Validate,
                                                           c2Validate) {}

  explicit BaseColorPacked4f(
      const uint8_t c0,
      const uint8_t c1,
      const uint8_t c2,
      const uint8_t opacity = 255,
      const C0ConvertToFloatChannel& c0ConvertToFloatChannel = C0ConvertToFloatChannel(),
      const C1ConvertToFloatChannel& c1ConvertToFloatChannel = C1ConvertToFloatChannel(),
      const C2ConvertToFloatChannel& c2ConvertToFloatChannel = C2ConvertToFloatChannel(),
      const C0Validator& c0Validate = C0Validator(),
      const C1Validator& c1Validate = C1Validator(),
      const C2Validator& c2Validate = C2Validator())
      : BaseColor4f<C0Validator, C1Validator, C2Validator>(
            c0ConvertToFloatChannel(c0),
            c1ConvertToFloatChannel(c1),
            c2ConvertToFloatChannel(c2),
            opacityConvertToFloatChannel(opacity),
            c0Validate,
            c1Validate,
            c2Validate) {}

  explicit BaseColorPacked4f(
      const std::array<uint8_t, 4>& color,
      const C0ConvertToFloatChannel& c0ConvertToFloatChannel = C0ConvertToFloatChannel(),
      const C1ConvertToFloatChannel& c1ConvertToFloatChannel = C1ConvertToFloatChannel(),
      const C2ConvertToFloatChannel& c2ConvertToFloatChannel = C2ConvertToFloatChannel(),
      const C0Validator& c0Validate = C0Validator(),
      const C1Validator& c1Validate = C1Validator(),
      const C2Validator& c2Validate = C2Validator())
      : BaseColor4f<C0Validator, C1Validator, C2Validator>(
            c0ConvertToFloatChannel(color[0]),
            c1ConvertToFloatChannel(color[1]),
            c2ConvertToFloatChannel(color[2]),
            opacityConvertToFloatChannel(color[3]),
            c0Validate,
            c1Validate,
            c2Validate) {}

  explicit BaseColorPacked4f(
      const uint32_t packedColor,
      const C0ConvertToFloatChannel& c0ConvertToFloatChannel = C0ConvertToFloatChannel(),
      const C1ConvertToFloatChannel& c1ConvertToFloatChannel = C1ConvertToFloatChannel(),
      const C2ConvertToFloatChannel& c2ConvertToFloatChannel = C2ConvertToFloatChannel(),
      const C0Validator& c0Validate = C0Validator(),
      const C1Validator& c1Validate = C1Validator(),
      const C2Validator& c2Validate = C2Validator())
      : BaseColor4f<C0Validator, C1Validator, C2Validator>(
            c0ConvertToFloatChannel(static_cast<uint8_t>((packedColor >> 24) & 0xFF)),
            c1ConvertToFloatChannel(static_cast<uint8_t>((packedColor >> 16) & 0xFF)),
            c2ConvertToFloatChannel(static_cast<uint8_t>((packedColor >> 8) & 0xFF)),
            opacityConvertToFloatChannel(static_cast<uint8_t>(packedColor & 0xFF)),
            c0Validate,
            c1Validate,
            c2Validate) {}

  ~BaseColorPacked4f() {}
  void setFromPackedComponents(
      const uint8_t c0,
      const uint8_t c1,
      const uint8_t c2,
      const uint8_t opacity = 255,
      const C0ConvertToFloatChannel& c0ConvertToFloatChannel = C0ConvertToFloatChannel(),
      const C1ConvertToFloatChannel& c1ConvertToFloatChannel = C1ConvertToFloatChannel(),
      const C2ConvertToFloatChannel& c2ConvertToFloatChannel = C2ConvertToFloatChannel(),
      const C0Validator& c0Validate = C0Validator(),
      const C1Validator& c1Validate = C1Validator(),
      const C2Validator& c2Validate = C2Validator()) {
    BaseColor4f<C0Validator, C1Validator, C2Validator>::set(
        c0ConvertToFloatChannel(c0),
        c1ConvertToFloatChannel(c1),
        c2ConvertToFloatChannel(c2),
        opacityConvertToFloatChannel(opacity),
        c0Validate,
        c1Validate,
        c2Validate);
  }

  void setFromPackedColor(
      const uint32_t packedColor,
      const C0ConvertToFloatChannel& c0ConvertToFloatChannel = C0ConvertToFloatChannel(),
      const C1ConvertToFloatChannel& c1ConvertToFloatChannel = C1ConvertToFloatChannel(),
      const C2ConvertToFloatChannel& c2ConvertToFloatChannel = C2ConvertToFloatChannel(),
      const C0Validator& c0Validate = C0Validator(),
      const C1Validator& c1Validate = C1Validator(),
      const C2Validator& c2Validate = C2Validator()) {
    BaseColor4f<C0Validator, C1Validator, C2Validator>::set(
        c0ConvertToFloatChannel(static_cast<uint8_t>((packedColor >> 24) & 0xFF)),
        c1ConvertToFloatChannel(static_cast<uint8_t>((packedColor >> 16) & 0xFF)),
        c2ConvertToFloatChannel(static_cast<uint8_t>((packedColor >> 8) & 0xFF)),
        opacityConvertToFloatChannel(static_cast<uint8_t>(packedColor & 0xFF)),
        c0Validate,
        c1Validate,
        c2Validate);
  }

  uint32_t getPackedColor(
      const C0ConvertToFloatChannel& c0ConvertToFloatChannel = C0ConvertToFloatChannel(),
      const C1ConvertToFloatChannel& c1ConvertToFloatChannel = C1ConvertToFloatChannel(),
      const C2ConvertToFloatChannel& c2ConvertToFloatChannel =
          C2ConvertToFloatChannel()) const {
    uint32_t rtn(0);
    auto& colorArray = BaseColor4f<C0Validator, C1Validator, C2Validator>::_colorArray;

    auto c0 = c0ConvertToFloatChannel.inverse(colorArray[0]);
    rtn |= (c0 << 24);
    auto c1 = c1ConvertToFloatChannel.inverse(colorArray[1]);
    rtn |= (c1 << 16);
    auto c2 = c2ConvertToFloatChannel.inverse(colorArray[2]);
    rtn |= (c2 << 8);
    auto opacity = opacityConvertToFloatChannel.inverse(colorArray[3]);
    rtn |= opacity;

    return rtn;
  }
};

}  // namespace gfx
